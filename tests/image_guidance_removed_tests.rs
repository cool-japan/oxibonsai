//! Regression test for finding image-01: `--guidance` / `:guidance` was a
//! fully inert CLI/REPL knob (Bonsai-Image is a distilled, CFG-free model
//! with `guidance_embeds: false`, so the value could never affect a single
//! output pixel) presented to users as a real parameter with no caveat.
//!
//! The fix removes the knob entirely (pre-1.0 breaking change, per the work
//! order) rather than keep dishonestly pretending it does something. These
//! tests prove:
//!   1. `--help` for `image` and `repl` no longer advertises `--guidance`.
//!   2. Passing `--guidance` is now a clap "unrecognized argument" error
//!      instead of being silently accepted and discarded.

use std::process::Command;

fn run_help(subcommand: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args([subcommand, "--help"])
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        output.status.success(),
        "`{subcommand} --help` should succeed; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn image_help_no_longer_advertises_guidance() {
    let stdout = run_help("image");
    assert!(
        !stdout.contains("guidance"),
        "`image --help` must not advertise the removed --guidance flag; got:\n{stdout}"
    );
}

#[test]
fn repl_help_no_longer_advertises_guidance() {
    let stdout = run_help("repl");
    assert!(
        !stdout.contains("guidance"),
        "`repl --help` must not advertise the removed --guidance flag; got:\n{stdout}"
    );
}

#[test]
fn image_rejects_unknown_guidance_flag() {
    // No real model assets are required to observe the clap parse error:
    // unrecognized-flag rejection happens before any model/asset resolution.
    let output = Command::new(env!("CARGO_BIN_EXE_oxibonsai"))
        .args([
            "image",
            "--prompt",
            "x",
            "--out",
            "/dev/null",
            "--guidance",
            "3.0",
        ])
        .output()
        .expect("failed to spawn oxibonsai binary");
    assert!(
        !output.status.success(),
        "`image --guidance` must fail now that the flag was removed"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unexpected argument") || stderr.contains("--guidance"),
        "expected a clap unrecognized-argument error mentioning --guidance; got:\n{stderr}"
    );
}
