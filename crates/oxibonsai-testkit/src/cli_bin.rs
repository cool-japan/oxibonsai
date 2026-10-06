//! Locating the `oxibonsai` command-line binary for tests that run it as a
//! subprocess.
//!
//! `oxibonsai info`, `oxibonsai eval` and friends live in a bin-only crate
//! (`oxibonsai-cli` has no `[lib]` target), so a test that wants to check
//! what the shipped command prints has to execute the shipped binary. This
//! module is the one place that decides *which* binary that is, so every such
//! test agrees and none of them compiles anything at a bad moment.
//!
//! # Why tests must not build the CLI themselves
//!
//! A test that runs a bare `cargo build --release -p oxibonsai-cli` and then
//! executes `<manifest>/../../target/release/oxibonsai` has three defects:
//!
//! 1. it compiles for minutes *inside* the real-model critical section (the
//!    lock the caller holds while a multi-GB model is mapped), so the
//!    compile time is billed to the gate;
//! 2. a build with a narrower feature set than the one already on disk
//!    *overwrites* the wider binary in a shared target directory, silently
//!    voiding every later leg that expected (say) the Metal backend;
//! 3. it runs a binary from a directory other than the one it built into
//!    whenever `CARGO_TARGET_DIR` is set, because the path is computed
//!    instead of reported by cargo.
//!
//! # The resolution rules
//!
//! [`resolve_cli_binary`] applies these in order and stops at the first that
//! applies:
//!
//! 1. **`OXIBONSAI_CLI_BIN`** ([`CLI_BIN_ENV`]) is set: that path is used. It
//!    must be an existing file and `<path> --version` must exit 0, otherwise
//!    the call fails with a typed [`CliBinError`]. It never falls through to a
//!    build, so a stale variable cannot hide a missing build. The release
//!    gate builds the CLI once, exports the variable, and every later stage
//!    (including the tests) reuses that one binary.
//! 2. **Otherwise** `cargo build --release -p oxibonsai-cli --bin oxibonsai
//!    --all-features --message-format=json` runs with the caller's
//!    environment (so `CARGO_TARGET_DIR` is honoured), and the executable
//!    path is the `executable` of the `compiler-artifact` record whose
//!    `target.name` is `oxibonsai` — the path cargo itself reports, never a
//!    computed `<manifest>/../../target`. The feature set is always the full
//!    one, so this build can only ever produce (or refresh) the widest
//!    binary, never replace it with a narrower one.
//!
//! A test calls [`resolve_cli_binary`] *before* it maps a real model or takes
//! a real-model lock, and reuses the returned path afterwards. The outcome of
//! the first successful resolution is cached for the process, so several
//! tests in one binary build at most once. Every resolution logs the path and
//! the rule that produced it to stderr.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

/// The environment variable that names a pre-built CLI binary (rule 1).
pub const CLI_BIN_ENV: &str = "OXIBONSAI_CLI_BIN";

/// The cargo package that owns the CLI binary.
pub const CLI_PACKAGE: &str = "oxibonsai-cli";

/// The CLI binary's target name (also the name of an unrelated library
/// crate, whose `compiler-artifact` record carries `executable: null`).
pub const CLI_BINARY: &str = "oxibonsai";

/// The exact arguments of the build in rule 2.
pub const CLI_BUILD_ARGS: [&str; 8] = [
    "build",
    "--release",
    "-p",
    CLI_PACKAGE,
    "--bin",
    CLI_BINARY,
    "--all-features",
    "--message-format=json",
];

/// How many times a `--version` probe is retried when the OS reports the
/// executable as busy (`ETXTBSY`): a script written moments ago can still be
/// held open for writing by a concurrent `fork` in another thread.
const BUSY_RETRIES: u32 = 10;

/// Which resolution rule produced a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliBinRule {
    /// Rule 1: the path named by `OXIBONSAI_CLI_BIN`.
    EnvOverride,
    /// Rule 2: the executable cargo reported for a fresh release build.
    CargoBuild,
}

impl fmt::Display for CliBinRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EnvOverride => f.write_str("the OXIBONSAI_CLI_BIN override (rule 1)"),
            Self::CargoBuild => f.write_str("`cargo build --release --all-features` (rule 2)"),
        }
    }
}

/// A resolved CLI binary and the rule that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCli {
    /// The executable to run.
    pub path: PathBuf,
    /// How it was found.
    pub rule: CliBinRule,
}

/// Why the CLI binary could not be resolved.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CliBinError {
    /// `OXIBONSAI_CLI_BIN` names something that is not an existing file.
    #[error(
        "OXIBONSAI_CLI_BIN is set to {path:?}, which is not an existing file \
         (unset the variable to build the CLI instead)"
    )]
    OverrideMissing {
        /// The value of the variable.
        path: PathBuf,
    },
    /// The binary exists but could not be started.
    #[error("the CLI binary {path:?} (from {rule}) could not be run: {error}")]
    NotRunnable {
        /// The binary.
        path: PathBuf,
        /// The rule that produced it.
        rule: CliBinRule,
        /// The operating system's reason.
        error: String,
    },
    /// `<binary> --version` did not exit 0.
    #[error("`{path:?} --version` (binary from {rule}) failed with {status}; stderr: {stderr}")]
    VersionFailed {
        /// The binary.
        path: PathBuf,
        /// The rule that produced it.
        rule: CliBinRule,
        /// The exit status, rendered.
        status: String,
        /// The probe's stderr, trimmed.
        stderr: String,
    },
    /// `cargo` could not be started.
    #[error("could not start `{program} build` for the CLI: {error}")]
    BuildSpawn {
        /// The cargo program that was run.
        program: String,
        /// The operating system's reason.
        error: String,
    },
    /// The build ran and failed; cargo's own diagnostics went to stderr.
    #[error("building the CLI failed ({status}); cargo's diagnostics are in the output above")]
    BuildFailed {
        /// The exit status, rendered.
        status: String,
    },
    /// The build succeeded but its JSON stream names no executable for the
    /// CLI binary.
    #[error(
        "cargo's JSON stream has no `compiler-artifact` with an executable for the \
         `oxibonsai` binary of `oxibonsai-cli`"
    )]
    NoArtifact,
    /// Cargo named an executable that is not on disk.
    #[error("cargo reported the CLI executable at {path:?}, but no file exists there")]
    ArtifactMissing {
        /// The reported path.
        path: PathBuf,
    },
}

/// What a `cargo build` invocation produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildOutput {
    /// Whether cargo exited 0.
    pub success: bool,
    /// The exit status, rendered.
    pub status: String,
    /// Cargo's stdout: the `--message-format=json` stream.
    pub stdout: String,
}

/// The `executable` of the CLI binary in a `cargo build
/// --message-format=json` stream.
///
/// A record counts when its `reason` is `compiler-artifact`, its
/// `target.name` is `oxibonsai`, its `target.kind` (when present) contains
/// `bin`, and its `executable` is a non-null, non-empty string. The
/// `oxibonsai` *library* crate produces a record with the same target name
/// and `executable: null`; it, records for other targets, non-JSON lines and
/// every other `reason` are ignored. When several records qualify the last
/// one wins (cargo re-emits an unchanged artifact as fresh).
///
/// # Errors
///
/// [`CliBinError::NoArtifact`] when no record qualifies;
/// [`CliBinError::BuildFailed`] when the stream carries a
/// `build-finished` record with `success: false`.
pub fn executable_from_cargo_stream(stream: &str) -> Result<PathBuf, CliBinError> {
    let mut found = None;
    for line in stream.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Ok(record) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let build_failed =
            record.get("success").and_then(serde_json::Value::as_bool) == Some(false);
        match record.get("reason").and_then(serde_json::Value::as_str) {
            Some("compiler-artifact") => {
                if let Some(path) = cli_executable(&record) {
                    found = Some(path);
                }
            }
            Some("build-finished") if build_failed => {
                return Err(CliBinError::BuildFailed {
                    status: "cargo reported `success: false`".to_string(),
                });
            }
            _ => {}
        }
    }
    found.ok_or(CliBinError::NoArtifact)
}

/// The executable of one `compiler-artifact` record, if it is the CLI binary.
fn cli_executable(record: &serde_json::Value) -> Option<PathBuf> {
    let target = record.get("target")?;
    if target.get("name")?.as_str()? != CLI_BINARY {
        return None;
    }
    if let Some(kinds) = target.get("kind").and_then(serde_json::Value::as_array) {
        if !kinds.iter().any(|kind| kind.as_str() == Some("bin")) {
            return None;
        }
    }
    let executable = record.get("executable")?.as_str()?;
    if executable.is_empty() {
        None
    } else {
        Some(PathBuf::from(executable))
    }
}

/// The cargo program: `$CARGO` (which cargo sets for the tests it runs), else
/// `cargo` from `PATH`.
fn cargo_program() -> OsString {
    std::env::var_os("CARGO")
        .filter(|program| !program.is_empty())
        .unwrap_or_else(|| OsString::from("cargo"))
}

/// The rule-2 build command. It inherits the caller's whole environment —
/// nothing is cleared and `CARGO_TARGET_DIR` is neither read nor set — so
/// the build lands wherever the caller's cargo would put it. Cargo's stdout
/// is piped (the JSON stream); its stderr is inherited so compile progress
/// and errors stay visible.
fn build_command(program: &OsStr) -> Command {
    let mut command = Command::new(program);
    command
        .args(CLI_BUILD_ARGS)
        .current_dir(crate::workspace::root())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    command
}

/// Run the rule-2 build with `program` and collect its JSON stream.
fn run_cargo_build(program: &OsStr) -> Result<BuildOutput, CliBinError> {
    let output = build_command(program)
        .output()
        .map_err(|e| CliBinError::BuildSpawn {
            program: program.to_string_lossy().into_owned(),
            error: e.to_string(),
        })?;
    Ok(BuildOutput {
        success: output.status.success(),
        status: output.status.to_string(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
    })
}

/// Check that `path` starts and answers `--version` with exit 0.
fn verify_runnable(path: &Path, rule: CliBinRule) -> Result<(), CliBinError> {
    let mut attempts = 0;
    let output = loop {
        match Command::new(path)
            .arg("--version")
            .stdin(Stdio::null())
            .output()
        {
            Ok(output) => break output,
            Err(e)
                if e.kind() == std::io::ErrorKind::ExecutableFileBusy
                    && attempts < BUSY_RETRIES =>
            {
                attempts += 1;
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                return Err(CliBinError::NotRunnable {
                    path: path.to_path_buf(),
                    rule,
                    error: e.to_string(),
                })
            }
        }
    };
    if output.status.success() {
        Ok(())
    } else {
        Err(CliBinError::VersionFailed {
            path: path.to_path_buf(),
            rule,
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        })
    }
}

/// The rules, with their inputs passed in so they can be exercised without
/// touching the process environment or running cargo: `override_path` is the
/// value of `OXIBONSAI_CLI_BIN` (rule 1, when `Some`) and `build` performs
/// rule 2's build. `build` is never called when `override_path` is `Some`.
fn resolve_with(
    override_path: Option<OsString>,
    build: impl FnOnce() -> Result<BuildOutput, CliBinError>,
) -> Result<ResolvedCli, CliBinError> {
    if let Some(raw) = override_path {
        let path = PathBuf::from(raw);
        if !path.is_file() {
            return Err(CliBinError::OverrideMissing { path });
        }
        verify_runnable(&path, CliBinRule::EnvOverride)?;
        return Ok(ResolvedCli {
            path,
            rule: CliBinRule::EnvOverride,
        });
    }
    let built = build()?;
    if !built.success {
        return Err(CliBinError::BuildFailed {
            status: built.status,
        });
    }
    let path = executable_from_cargo_stream(&built.stdout)?;
    if !path.is_file() {
        return Err(CliBinError::ArtifactMissing { path });
    }
    verify_runnable(&path, CliBinRule::CargoBuild)?;
    Ok(ResolvedCli {
        path,
        rule: CliBinRule::CargoBuild,
    })
}

/// [`resolve_with`] fed from the real environment and the real cargo.
fn resolve_from_environment() -> Result<ResolvedCli, CliBinError> {
    let override_path = std::env::var_os(CLI_BIN_ENV);
    if override_path.is_none() {
        eprintln!(
            "cli_bin: {CLI_BIN_ENV} is not set; building the CLI with `cargo {}`",
            CLI_BUILD_ARGS.join(" ")
        );
    }
    resolve_with(override_path, || run_cargo_build(&cargo_program()))
}

/// The first successful resolution in this process.
static RESOLVED: Mutex<Option<ResolvedCli>> = Mutex::new(None);

/// Run `resolve` unless `cache` already holds a resolution, logging (to
/// stderr) the path and the rule of a fresh one and caching it. The lock is
/// held for the whole resolution, so a second caller waits for the first
/// instead of starting a second build. Failures are not cached.
fn resolve_cached(
    cache: &Mutex<Option<ResolvedCli>>,
    resolve: impl FnOnce() -> Result<ResolvedCli, CliBinError>,
) -> Result<ResolvedCli, CliBinError> {
    let mut cached = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(resolved) = cached.as_ref() {
        eprintln!(
            "cli_bin: resolved {:?} via {} (reused from earlier in this process)",
            resolved.path, resolved.rule
        );
        return Ok(resolved.clone());
    }
    let resolved = resolve()?;
    eprintln!(
        "cli_bin: resolved {:?} via {}",
        resolved.path, resolved.rule
    );
    *cached = Some(resolved.clone());
    Ok(resolved)
}

/// Resolve the CLI binary and report which rule produced it (see the module
/// docs for the rules). The first success is cached for the process, so
/// concurrent and repeated callers build at most once. Failures are not
/// cached.
///
/// # Errors
///
/// A [`CliBinError`] naming what failed: an unusable override, a failed
/// build, or a build whose output has no executable for the CLI.
pub fn resolve_cli_binary_with_rule() -> Result<ResolvedCli, CliBinError> {
    resolve_cached(&RESOLVED, resolve_from_environment)
}

/// The path of the CLI binary tests should run (see the module docs).
///
/// Resolve it before mapping a real model or taking a real-model lock, then
/// run it:
///
/// ```no_run
/// // `no_run`: resolving may run a multi-minute release build.
/// let bin = oxibonsai_testkit::cli_bin::resolve_cli_binary()
///     .expect("the oxibonsai CLI binary resolves");
/// let output = std::process::Command::new(bin)
///     .arg("--version")
///     .output()
///     .expect("the CLI starts");
/// assert!(output.status.success());
/// ```
///
/// # Errors
///
/// See [`resolve_cli_binary_with_rule`].
pub fn resolve_cli_binary() -> Result<PathBuf, CliBinError> {
    resolve_cli_binary_with_rule().map(|resolved| resolved.path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::Mutex as StdMutex;

    /// Guards the one test that sets `OXIBONSAI_CLI_BIN` in the process
    /// environment (`cargo test` runs a crate's tests on parallel threads).
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    /// A `compiler-artifact` record for a target of the given kind.
    fn artifact(name: &str, kind: &str, executable: Option<&str>) -> String {
        serde_json::json!({
            "reason": "compiler-artifact",
            "package_id": format!("path+file:///workspace#{name}@0.0.0"),
            "target": { "kind": [kind], "crate_types": [kind], "name": name },
            "profile": { "opt_level": "3" },
            "features": ["default"],
            "filenames": [],
            "executable": executable,
            "fresh": false,
        })
        .to_string()
    }

    fn finished(success: bool) -> String {
        serde_json::json!({ "reason": "build-finished", "success": success }).to_string()
    }

    #[test]
    fn stream_picks_the_cli_binary_among_several_artifacts() {
        let stream = [
            artifact("serde", "lib", None),
            // The facade crate is a library named like the binary.
            artifact("oxibonsai", "lib", None),
            artifact(
                "oxibonsai-serve",
                "bin",
                Some("target/release/oxibonsai-serve"),
            ),
            // An example with the same target name is not the binary.
            artifact(
                "oxibonsai",
                "example",
                Some("target/release/examples/oxibonsai"),
            ),
            artifact("oxibonsai", "bin", Some("target/release/oxibonsai")),
            finished(true),
        ]
        .join("\n");
        assert_eq!(
            executable_from_cargo_stream(&stream).ok(),
            Some(PathBuf::from("target/release/oxibonsai"))
        );
    }

    #[test]
    fn stream_ignores_records_whose_executable_is_null() {
        let stream = [
            artifact("oxibonsai", "lib", None),
            artifact("oxibonsai", "bin", None),
            finished(true),
        ]
        .join("\n");
        assert!(matches!(
            executable_from_cargo_stream(&stream),
            Err(CliBinError::NoArtifact)
        ));
    }

    #[test]
    fn stream_without_a_matching_artifact_is_an_error() {
        for stream in [
            String::new(),
            "   \n\n".to_string(),
            [
                artifact("oxibonsai-runtime", "lib", None),
                artifact("oxibonsai-serve", "bin", Some("t/oxibonsai-serve")),
            ]
            .join("\n"),
        ] {
            assert!(
                matches!(
                    executable_from_cargo_stream(&stream),
                    Err(CliBinError::NoArtifact)
                ),
                "stream {stream:?} must yield NoArtifact"
            );
        }
    }

    #[test]
    fn stream_skips_non_json_lines_and_empty_executables() {
        let stream = [
            "   Compiling oxibonsai-cli v0.0.0".to_string(),
            "{not json".to_string(),
            r#"{"reason":"compiler-message","message":{}}"#.to_string(),
            r#"{"target":{"name":"oxibonsai","kind":["bin"]},"executable":"t/no-reason"}"#
                .to_string(),
            artifact("oxibonsai", "bin", Some("")),
            artifact("oxibonsai", "bin", Some("t/oxibonsai")),
        ]
        .join("\n");
        assert_eq!(
            executable_from_cargo_stream(&stream).ok(),
            Some(PathBuf::from("t/oxibonsai"))
        );
    }

    #[test]
    fn stream_takes_the_last_of_several_matching_records() {
        let stream = [
            artifact("oxibonsai", "bin", Some("t/first/oxibonsai")),
            artifact("oxibonsai", "bin", Some("t/second/oxibonsai")),
        ]
        .join("\n");
        assert_eq!(
            executable_from_cargo_stream(&stream).ok(),
            Some(PathBuf::from("t/second/oxibonsai"))
        );
    }

    #[test]
    fn stream_reporting_a_failed_build_is_an_error() {
        let stream = [
            artifact("oxibonsai", "bin", Some("t/oxibonsai")),
            finished(false),
        ]
        .join("\n");
        assert!(matches!(
            executable_from_cargo_stream(&stream),
            Err(CliBinError::BuildFailed { .. })
        ));
    }

    #[test]
    fn build_command_uses_the_full_feature_set_and_leaves_the_environment_alone() {
        let command = build_command(OsStr::new("cargo"));
        let args: Vec<&OsStr> = command.get_args().collect();
        assert_eq!(
            args,
            [
                "build",
                "--release",
                "-p",
                "oxibonsai-cli",
                "--bin",
                "oxibonsai",
                "--all-features",
                "--message-format=json",
            ]
            .map(OsStr::new)
        );
        // No variable is set, removed or cleared, so the child inherits the
        // caller's `CARGO_TARGET_DIR` (and everything else) untouched.
        assert_eq!(command.get_envs().count(), 0);
        assert!(
            !args.iter().any(|a| {
                let a = a.to_string_lossy();
                a.contains("target-dir") || a == "--features" || a == "--no-default-features"
            }),
            "the build must never narrow the feature set or move the target directory"
        );
    }

    #[test]
    fn env_and_constants_have_the_documented_spelling() {
        assert_eq!(CLI_BIN_ENV, "OXIBONSAI_CLI_BIN");
        assert_eq!(CLI_PACKAGE, "oxibonsai-cli");
        assert_eq!(CLI_BINARY, "oxibonsai");
    }

    /// A build that must never run: flags itself when called.
    struct BuildProbe {
        called: Cell<bool>,
    }

    impl BuildProbe {
        fn new() -> Self {
            Self {
                called: Cell::new(false),
            }
        }

        fn run(&self) -> Result<BuildOutput, CliBinError> {
            self.called.set(true);
            Ok(BuildOutput {
                success: true,
                status: "exit status: 0".to_string(),
                stdout: String::new(),
            })
        }
    }

    fn resolved(path: &str) -> ResolvedCli {
        ResolvedCli {
            path: PathBuf::from(path),
            rule: CliBinRule::CargoBuild,
        }
    }

    #[test]
    fn a_successful_resolution_is_cached_and_a_failure_is_not() {
        let cache = Mutex::new(None);
        let calls = Cell::new(0u32);
        let first = resolve_cached(&cache, || {
            calls.set(calls.get() + 1);
            Err(CliBinError::NoArtifact)
        });
        assert!(matches!(first, Err(CliBinError::NoArtifact)));
        // The failure was not cached: the next call resolves again.
        let second = resolve_cached(&cache, || {
            calls.set(calls.get() + 1);
            Ok(resolved("t/oxibonsai"))
        });
        assert_eq!(second.ok(), Some(resolved("t/oxibonsai")));
        // The success was: no further resolution, whatever the closure says.
        let third = resolve_cached(&cache, || {
            calls.set(calls.get() + 1);
            Err(CliBinError::NoArtifact)
        });
        assert_eq!(third.ok(), Some(resolved("t/oxibonsai")));
        assert_eq!(calls.get(), 2, "one failed and one successful resolution");
    }

    #[test]
    fn concurrent_callers_share_one_resolution() {
        let cache = Mutex::new(None);
        let calls = std::sync::atomic::AtomicU32::new(0);
        let results: Vec<Option<ResolvedCli>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        resolve_cached(&cache, || {
                            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            // Long enough for the other callers to queue up.
                            std::thread::sleep(Duration::from_millis(50));
                            Ok(resolved("t/oxibonsai"))
                        })
                        .ok()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap_or(None))
                .collect()
        });
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the build must run once for all callers"
        );
        assert!(results
            .iter()
            .all(|r| r.as_ref() == Some(&resolved("t/oxibonsai"))));
    }

    #[test]
    fn a_missing_override_is_a_typed_error_and_never_builds() {
        let dir = crate::temp_path::unique_path("cli-bin-missing", "");
        let missing = dir.join("no-such-binary");
        let probe = BuildProbe::new();
        let result = resolve_with(Some(missing.clone().into_os_string()), || probe.run());
        assert!(
            matches!(&result, Err(CliBinError::OverrideMissing { path }) if *path == missing),
            "got {result:?}"
        );
        assert!(
            !probe.called.get(),
            "a stale OXIBONSAI_CLI_BIN must not fall through to a build"
        );
        // An empty value is still "set": it names no file, so it is an error too.
        let probe = BuildProbe::new();
        let result = resolve_with(Some(OsString::new()), || probe.run());
        assert!(matches!(result, Err(CliBinError::OverrideMissing { .. })));
        assert!(!probe.called.get());
    }

    #[cfg(unix)]
    mod unix {
        use super::*;
        use std::os::unix::fs::PermissionsExt;

        /// A scratch directory removed on drop.
        struct Scratch(PathBuf);

        impl Scratch {
            fn new(tag: &str) -> Self {
                let dir = crate::temp_path::unique_path(tag, "");
                std::fs::create_dir_all(&dir)
                    .unwrap_or_else(|e| panic!("create scratch dir {dir:?}: {e}"));
                Self(dir)
            }

            fn script(&self, name: &str, body: &str, mode: u32) -> PathBuf {
                let path = self.0.join(name);
                std::fs::write(&path, format!("#!/bin/sh\n{body}\n"))
                    .unwrap_or_else(|e| panic!("write {path:?}: {e}"));
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode))
                    .unwrap_or_else(|e| panic!("chmod {path:?}: {e}"));
                path
            }
        }

        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        #[test]
        fn a_runnable_override_wins_over_the_build() {
            let scratch = Scratch::new("cli-bin-override-ok");
            let cli = scratch.script("oxibonsai", "exit 0", 0o755);
            let probe = BuildProbe::new();
            let resolved = resolve_with(Some(cli.clone().into_os_string()), || probe.run())
                .unwrap_or_else(|e| panic!("a runnable override must resolve: {e}"));
            assert_eq!(
                resolved,
                ResolvedCli {
                    path: cli,
                    rule: CliBinRule::EnvOverride
                }
            );
            assert!(!probe.called.get(), "the override must pre-empt the build");
        }

        #[test]
        fn an_override_whose_version_probe_fails_is_a_typed_error() {
            let scratch = Scratch::new("cli-bin-override-version");
            let cli = scratch.script("oxibonsai", "echo boom >&2\nexit 3", 0o755);
            let probe = BuildProbe::new();
            let result = resolve_with(Some(cli.clone().into_os_string()), || probe.run());
            match result {
                Err(CliBinError::VersionFailed {
                    path,
                    rule,
                    status,
                    stderr,
                }) => {
                    assert_eq!(path, cli);
                    assert_eq!(rule, CliBinRule::EnvOverride);
                    assert!(status.contains('3'), "status {status:?}");
                    assert_eq!(stderr, "boom");
                }
                other => panic!("expected VersionFailed, got {other:?}"),
            }
            assert!(!probe.called.get());
        }

        #[test]
        fn an_override_that_cannot_be_executed_is_a_typed_error() {
            let scratch = Scratch::new("cli-bin-override-noexec");
            let cli = scratch.script("oxibonsai", "exit 0", 0o644);
            let probe = BuildProbe::new();
            let result = resolve_with(Some(cli.into_os_string()), || probe.run());
            assert!(
                matches!(result, Err(CliBinError::NotRunnable { .. })),
                "got {result:?}"
            );
            assert!(!probe.called.get());
        }

        #[test]
        fn without_an_override_the_reported_artifact_is_used() {
            let scratch = Scratch::new("cli-bin-build-ok");
            let cli = scratch.script("oxibonsai", "exit 0", 0o755);
            let stdout = [
                artifact("oxibonsai", "lib", None),
                artifact("oxibonsai", "bin", Some(&cli.to_string_lossy())),
                finished(true),
            ]
            .join("\n");
            let resolved = resolve_with(None, || {
                Ok(BuildOutput {
                    success: true,
                    status: "exit status: 0".to_string(),
                    stdout,
                })
            })
            .unwrap_or_else(|e| panic!("a reported, runnable artifact must resolve: {e}"));
            assert_eq!(
                resolved,
                ResolvedCli {
                    path: cli,
                    rule: CliBinRule::CargoBuild
                }
            );
        }

        #[test]
        fn a_failed_or_artifactless_build_is_a_typed_error() {
            let failed = resolve_with(None, || {
                Ok(BuildOutput {
                    success: false,
                    status: "exit status: 101".to_string(),
                    stdout: String::new(),
                })
            });
            assert!(
                matches!(&failed, Err(CliBinError::BuildFailed { status }) if status.contains("101")),
                "got {failed:?}"
            );
            let no_artifact = resolve_with(None, || {
                Ok(BuildOutput {
                    success: true,
                    status: "exit status: 0".to_string(),
                    stdout: artifact("oxibonsai", "lib", None),
                })
            });
            assert!(
                matches!(no_artifact, Err(CliBinError::NoArtifact)),
                "got {no_artifact:?}"
            );
            let scratch = Scratch::new("cli-bin-build-gone");
            let gone = scratch.0.join("deleted").join("oxibonsai");
            let stale = resolve_with(None, || {
                Ok(BuildOutput {
                    success: true,
                    status: "exit status: 0".to_string(),
                    stdout: artifact("oxibonsai", "bin", Some(&gone.to_string_lossy())),
                })
            });
            assert!(
                matches!(&stale, Err(CliBinError::ArtifactMissing { path }) if *path == gone),
                "got {stale:?}"
            );
        }

        #[test]
        fn a_built_binary_that_fails_its_version_probe_is_rejected() {
            let scratch = Scratch::new("cli-bin-build-badversion");
            let cli = scratch.script("oxibonsai", "exit 2", 0o755);
            let result = resolve_with(None, || {
                Ok(BuildOutput {
                    success: true,
                    status: "exit status: 0".to_string(),
                    stdout: artifact("oxibonsai", "bin", Some(&cli.to_string_lossy())),
                })
            });
            assert!(
                matches!(
                    result,
                    Err(CliBinError::VersionFailed {
                        rule: CliBinRule::CargoBuild,
                        ..
                    })
                ),
                "got {result:?}"
            );
        }

        /// The whole of rule 2 against a stand-in `cargo`: it receives
        /// exactly the documented arguments, its JSON stream is parsed for
        /// the CLI binary, and a non-zero exit is a failed build.
        #[test]
        fn the_build_runs_the_documented_command_through_a_stand_in_cargo() {
            let scratch = Scratch::new("cli-bin-fake-cargo");
            let cli = scratch.script("oxibonsai", "exit 0", 0o755);
            let args_log = scratch.0.join("args.txt");
            let stream = [
                artifact("oxibonsai", "lib", None),
                artifact("oxibonsai", "bin", Some(&cli.to_string_lossy())),
                finished(true),
            ]
            .join("\n");
            let stream_file = scratch.0.join("stream.jsonl");
            std::fs::write(&stream_file, stream)
                .unwrap_or_else(|e| panic!("write the canned stream: {e}"));
            let cargo = scratch.script(
                "cargo",
                &format!(
                    "printf '%s\\n' \"$@\" > '{}'\ncat '{}'",
                    args_log.display(),
                    stream_file.display()
                ),
                0o755,
            );

            let resolved = resolve_with(None, || run_cargo_build(cargo.as_os_str()))
                .unwrap_or_else(|e| panic!("the stand-in cargo build must resolve: {e}"));
            assert_eq!(resolved.path, cli);
            assert_eq!(resolved.rule, CliBinRule::CargoBuild);
            let logged = std::fs::read_to_string(&args_log)
                .unwrap_or_else(|e| panic!("read the recorded arguments: {e}"));
            assert_eq!(
                logged.lines().collect::<Vec<_>>(),
                CLI_BUILD_ARGS.to_vec(),
                "cargo must receive exactly the documented arguments"
            );

            let failing = scratch.script(
                "cargo-fails",
                "echo 'stand-in cargo: simulated build failure (expected by this test)' >&2\nexit 101",
                0o755,
            );
            let result = resolve_with(None, || run_cargo_build(failing.as_os_str()));
            assert!(
                matches!(&result, Err(CliBinError::BuildFailed { status }) if status.contains("101")),
                "got {result:?}"
            );

            let absent = scratch.0.join("no-such-cargo");
            let result = resolve_with(None, || run_cargo_build(absent.as_os_str()));
            assert!(
                matches!(result, Err(CliBinError::BuildSpawn { .. })),
                "got {result:?}"
            );
        }

        /// The environment is read exactly once per resolution, by the
        /// real-environment entry point (the pure rules above never touch it).
        #[test]
        fn the_environment_variable_selects_the_override() {
            let _guard = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let scratch = Scratch::new("cli-bin-env");
            let cli = scratch.script("oxibonsai", "exit 0", 0o755);
            let prior = std::env::var_os(CLI_BIN_ENV);
            std::env::set_var(CLI_BIN_ENV, &cli);
            let resolved = resolve_from_environment();
            match prior {
                Some(value) => std::env::set_var(CLI_BIN_ENV, value),
                None => std::env::remove_var(CLI_BIN_ENV),
            }
            let resolved =
                resolved.unwrap_or_else(|e| panic!("the override in the environment: {e}"));
            assert_eq!(resolved.path, cli);
            assert_eq!(resolved.rule, CliBinRule::EnvOverride);
        }
    }
}
