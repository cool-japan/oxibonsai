//! Hardware/fixture-capability self-skip reporting (T-05).
//!
//! Historically, every hardware-dependent test in this workspace self-skipped
//! to **green**: a test that detected "no Metal GPU" / "no CUDA device" /
//! "fixture file missing" simply `return`ed early from inside the test body,
//! which a passing `cargo nextest` exit code cannot distinguish from the test
//! having actually validated the capability it claims to. This module is the
//! *producer* half of the fix; `scripts/release-gate.sh` is the *consumer*
//! half that fails the release gate when a required capability has zero
//! fresh `executed: true` evidence.
//!
//! # The contract (read `scripts/release-gate.sh`'s header before changing this file)
//!
//! Path: the absolute path in the `OXIBONSAI_CAPABILITY_REPORT` environment
//! variable when the release gate set it; otherwise resolved locally (see
//! [`report_path`]).
//!
//! Format: JSON Lines — one compact JSON object per line, **appended**
//! (never rewritten), because `cargo nextest` runs test binaries as separate
//! OS processes and a shared JSON array would need a read-modify-write cycle
//! concurrent processes would corrupt. Every write is a single `write`
//! syscall of one complete, newline-terminated line.
//!
//! Each line: `{"capability": "metal", "executed": true, "test":
//! "oxibonsai-kernels::metal_k_quant_gemv_parity::metal_gemv_q2k_matches_scalar"}`,
//! optionally with a trailing `"duration_ms": <integer>` ([`record_timed`] /
//! [`record_executed_timed`]) — the record's own measured wall-clock cost,
//! omitted entirely (not `null`) by every call site that has not opted into
//! measuring it, so an older or unmeasured record stays exactly the shape
//! above. `executed: false` means the test body actually reached and took
//! the self-skip path (hardware/fixture absent) — it must still write a
//! record, so an all-skipped run is visibly distinct in the manifest from a
//! manifest that is simply missing (meaning no producer test uses this
//! contract yet).
//!
//! # THE INVARIANT
//!
//! **`executed: true` means the work the record names has already finished in
//! this process — never merely that the hardware or fixture it needs was
//! found to be present.** A presence probe is `executed: false`; the record
//! that claims execution belongs *after* the assertions, on the path only a
//! completed run can reach. Writing it up front is the exact false-green this
//! whole module exists to eliminate: the test can still panic afterwards, and
//! the manifest would go on asserting that the capability was validated.
//! [`record_executed`] and [`record_skipped`] exist so the right call is the
//! shorter one to write; [`record`] stays for call sites that compute the
//! flag.
//!
//! # Cross-crate wiring
//!
//! This is the canonical implementation: `oxibonsai-testkit` is a real
//! workspace member (root `Cargo.toml`'s `[workspace] members`) taken as a
//! `[dev-dependencies]` entry by every producer crate, so no producer test
//! file carries its own inline copy of [`record`]/[`report_path`].
//!
//! # The `test` field's naming scheme
//!
//! Every record's `test` field is `<cargo package>::<test target>::<fn>`:
//! `<cargo package>` is the crate name as `cargo test -p` spells it (e.g.
//! `oxibonsai-model`, `oxibonsai-runtime`, `oxibonsai-cli`); `<test target>`
//! is the integration-test binary's own file stem (e.g.
//! `hybrid_forward_parity_tests`, `bonsai2_engine_tests`) for a case that
//! lives under a crate's `tests/` directory, `lib` for a `#[test]` inside a
//! library crate's own `src/` (reached via `cargo test --lib`), or `bin` for
//! one inside a binary crate's `src/main.rs` (`cargo test --bin <name>`);
//! `<fn>` is the test function's own **bare name only** — never the `mod`
//! path nextest would print ahead of it for a test pulled in via `#[path]
//! mod …;` (e.g. a case in `bonsai2_real/greedy_gates.rs`, wired into
//! `hybrid_forward_parity_tests.rs` as `mod greedy_gates;`, nextest lists as
//! `greedy_gates::hybrid_real_27b_pq2_0_…`, but records here as
//! `oxibonsai-model::hybrid_forward_parity_tests::hybrid_real_27b_pq2_0_…`
//! — no `greedy_gates::` segment). Every producer in this workspace follows
//! the bare-name form, confirmed against this project's own real-27B
//! capability manifest; `scripts/release-gate.sh`'s `--require-tests` names
//! are written the same way, and a `mod`-qualified name would never match.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// One capability this workspace's release gate can enforce.
///
/// Extend this list in the same edit that adds a new gated capability, and
/// update `scripts/release-gate.sh`'s header comment to match — that script
/// owns the consumer side of this contract and is the single source of
/// truth for which capability names it currently checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Capability {
    /// A macOS host with a Metal-capable GPU device.
    Metal,
    /// A host with an accessible CUDA device (this workspace has none; see
    /// `CONTEXT.md` — CUDA is compile-blind on this session's machine).
    Cuda,
    /// The RAG real-generation fixture (a real tokenizer file) is present.
    RagRealGeneration,
    /// The image crate's parity-GGUF fixture / real-model file is present.
    ImageParity,
    /// The real, multi-GB legacy model GGUFs (+ tokenizer) are present under
    /// `models/`: this is about *model-file* presence, not Metal/CUDA
    /// hardware. The two
    /// `legacy_parity_tests.rs` files (model + runtime) used to record their
    /// missing-model self-skip under [`Self::Metal`], which is always
    /// already satisfied elsewhere on any Metal host (e.g.
    /// `metal_k_quant_gemv_parity.rs` writes 30+ `metal`/`executed=true`
    /// records regardless of these models), so a release-gate check that
    /// requires `metal` executed=true can never actually catch "the legacy
    /// real-model parity gate never ran". A distinct name lets a future
    /// `REQUIRED_CAPS` entry gate on this specifically.
    LegacyModels,
    /// The real Bonsai 2 27B GGUFs (`Ternary-Bonsai-2-27B-{PQ2_0,PTQ1_0}.gguf`)
    /// are present under `models/` (or `$OXIBONSAI_MODELS_DIR`) — the same
    /// "model-file presence, not hardware" shape as [`Self::LegacyModels`],
    /// and deliberately a distinct name for the same reason: any Metal host
    /// already satisfies [`Self::Metal`] regardless of whether these specific
    /// 27B gates ever ran, so `REQUIRED_CAPS` needs its own name to gate on
    /// them. `crates/oxibonsai-model/tests/bonsai2_real/harness.rs` and
    /// `crates/oxibonsai-runtime/tests/bonsai2_engine_tests.rs` both record
    /// through this variant rather than writing the `"bonsai2-models"` JSONL
    /// line by hand.
    Bonsai2Models,
}

impl Capability {
    /// The stable string this capability is recorded under. Must match
    /// `scripts/release-gate.sh`'s own list exactly.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::RagRealGeneration => "rag-real-generation",
            Self::ImageParity => "image-parity",
            Self::LegacyModels => "legacy-models",
            Self::Bonsai2Models => "bonsai2-models",
        }
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One JSONL record, matching `scripts/release-gate.sh`'s documented schema.
/// `capability`/`executed`/`test` are the original, required fields (field
/// names and order are part of the contract); `duration_ms` is a later,
/// OPTIONAL addition (`skip_serializing_if`, so a record written by this
/// exact shape is byte-identical to the pre-`duration_ms` format when no
/// duration is given) — a record's own measured wall-clock cost, for a
/// human reading the manifest to see which real-model gates dominate a
/// release-gate run without re-running it under a stopwatch.
#[derive(Debug, Clone, Serialize)]
struct CapabilityRecord<'a> {
    capability: &'a str,
    executed: bool,
    test: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u64>,
}

/// Resolve the capability-manifest path per the contract in
/// `scripts/release-gate.sh`:
///
/// 1. `OXIBONSAI_CAPABILITY_REPORT`, if set to a non-empty value — both gate
///    scripts export this as an absolute path before running anything that
///    might produce a record, so a producer invoked from either script
///    always takes this branch.
/// 2. `CARGO_TARGET_DIR`, if set to a non-empty value — a producer run by
///    hand (e.g. `CARGO_TARGET_DIR=$(pwd)/target cargo test ...`, this
///    session's own worktree convention) still lands in the right place.
/// 3. A path relative to *this crate's own* `CARGO_MANIFEST_DIR`
///    (`../../target/capability-report.json`). This is stable regardless of
///    which crate's test binary calls in, because `CARGO_MANIFEST_DIR` is
///    captured at `oxibonsai-testkit`'s own compile time (fixed at
///    `<workspace-root>/crates/oxibonsai-testkit`), not the caller's — and
///    every workspace member lives exactly two directories below the
///    workspace root as `crates/<name>`, so `../../target` always resolves
///    to the default `target/` at the workspace root.
#[must_use]
pub fn report_path() -> PathBuf {
    if let Ok(p) = std::env::var("OXIBONSAI_CAPABILITY_REPORT") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("capability-report.json");
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target")
        .join("capability-report.json")
}

/// Append one capability record to `path`.
///
/// Never panics: a producer test's pass/fail must depend on the capability
/// it exercises, not on this bookkeeping write, so an I/O error here is
/// logged to stderr and swallowed rather than propagated.
fn record_at(
    path: &Path,
    capability: Capability,
    executed: bool,
    test_name: &str,
    duration_ms: Option<u64>,
) {
    let record = CapabilityRecord {
        capability: capability.as_str(),
        executed,
        test: test_name,
        duration_ms,
    };
    let line = match serde_json::to_string(&record) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("testkit::capability::record: failed to serialize record: {e}");
            return;
        }
    };
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("testkit::capability::record: failed to create {parent:?}: {e}");
                return;
            }
        }
    }
    let mut file = match OpenOptions::new().create(true).append(true).open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("testkit::capability::record: failed to open {path:?}: {e}");
            return;
        }
    };
    // `writeln!(file, "{line}")` on a
    // `std::fs::File` performs TWO separate `write(2)` syscalls (the payload,
    // then `"\n"`). Concurrent `nextest` processes/threads all appending to
    // the same manifest can then interleave mid-record, producing malformed
    // JSONL lines this module's own doc comment (above) documents as
    // impossible ("every write is a single `write` syscall of one complete,
    // newline-terminated line"). Build ONE buffer containing the trailing
    // newline and issue exactly ONE `write_all` so the two-syscall race
    // cannot happen: `O_APPEND` guarantees a single `write(2)` is atomic
    // with respect to other writers on the same file, but two separate
    // writes are not atomic *as a pair*.
    let mut line = line;
    line.push('\n');
    if let Err(e) = file.write_all(line.as_bytes()) {
        eprintln!("testkit::capability::record: failed to write to {path:?}: {e}");
    }
}

/// Append one capability record to the manifest at [`report_path`].
///
/// Call this from *every* branch of a hardware/fixture-gated test — both the
/// self-skip branch (`executed: false`) and the branch that actually
/// exercised the capability (`executed: true`) — so the manifest always
/// distinguishes "ran and validated" from "skipped" from "never
/// instrumented at all" (an empty manifest).
///
/// **The invariant, in one sentence: `executed: true` means the work this
/// record names has already completed in this process, never that the
/// hardware or fixture it needs was merely found to be present.** Prefer
/// [`record_executed`] / [`record_skipped`], which say which branch they are
/// on at the call site; use this form only where the flag is genuinely
/// computed.
///
/// `test_name` should be the fully-qualified test name, e.g.
/// `"oxibonsai-kernels::metal_k_quant_gemv_parity::metal_gemv_q2k_matches_scalar"`.
pub fn record(capability: Capability, executed: bool, test_name: &str) {
    record_at(&report_path(), capability, executed, test_name, None);
}

/// [`record`], additionally attaching `duration` (rounded to the
/// millisecond) as the record's `duration_ms` field — the wall-clock cost
/// of the work this record names, so a human reading the manifest sees
/// which real-model gates dominate a release-gate run without re-running it
/// under a stopwatch. Every call site that already used [`record`] keeps
/// compiling unchanged (this is a new, additive function, not a signature
/// change to the existing ones), so this is opt-in per call site.
pub fn record_timed(
    capability: Capability,
    executed: bool,
    test_name: &str,
    duration: std::time::Duration,
) {
    record_at(
        &report_path(),
        capability,
        executed,
        test_name,
        Some(duration.as_millis().min(u128::from(u64::MAX)) as u64),
    );
}

/// Record that `test_name` **actually exercised** `capability` this run.
///
/// Call this only from a point the test reaches after the assertions that
/// validate the capability have run — typically the last statement of the
/// test body, or of the helper that owns the whole matrix. Calling it on
/// fixture presence, before the work, writes a claim the run may never make
/// good on; see this module's THE INVARIANT section.
///
/// ```no_run
/// // `no_run`: this APPENDS to the real capability manifest, and a doctest
/// // that actually ran it would write a false `executed: true` record into
/// // the very evidence file the release gate reads.
/// use oxibonsai_testkit::capability::{record_executed, Capability};
///
/// fn parity_gate() {
///     // ... every assertion the capability's evidence consists of ...
///     record_executed(Capability::Metal, "mycrate::myfile::parity_gate");
/// }
/// # parity_gate();
/// ```
pub fn record_executed(capability: Capability, test_name: &str) {
    record(capability, true, test_name);
}

/// [`record_executed`], additionally attaching `duration` as the record's
/// `duration_ms` field (see [`record_timed`]). For a real-model gate whose
/// own wall-clock cost is worth surfacing in the capability report without
/// re-running it under a stopwatch — measure from just before the real work
/// starts (after locating/loading a required fixture, not before) to the
/// call site itself.
///
/// ```no_run
/// // `no_run`: see `record_executed`'s own example — a doctest must never
/// // append to the real capability manifest.
/// use std::time::Instant;
///
/// use oxibonsai_testkit::capability::{record_executed_timed, Capability};
///
/// fn parity_gate() {
///     let start = Instant::now();
///     // ... every assertion the capability's evidence consists of ...
///     record_executed_timed(Capability::Metal, "mycrate::myfile::parity_gate", start.elapsed());
/// }
/// # parity_gate();
/// ```
pub fn record_executed_timed(
    capability: Capability,
    test_name: &str,
    duration: std::time::Duration,
) {
    record_timed(capability, true, test_name, duration);
}

/// Record that `test_name` **self-skipped**: the hardware or fixture
/// `capability` names was absent, so nothing was validated.
///
/// A skip must still write a record — a skipped run has to be visibly
/// distinct in the manifest both from a validated one and from a manifest
/// that no producer has ever written to.
///
/// ```no_run
/// // `no_run`: see [`record_executed`]'s example — a doctest must never
/// // append to the real capability manifest.
/// use oxibonsai_testkit::capability::{record_skipped, Capability};
///
/// fn parity_gate(fixture_present: bool) {
///     if !fixture_present {
///         record_skipped(Capability::Metal, "mycrate::myfile::parity_gate");
///         return;
///     }
///     // ... the real gate ...
/// }
/// # parity_gate(false);
/// ```
pub fn record_skipped(capability: Capability, test_name: &str) {
    record(capability, false, test_name);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Guards every test that mutates process-global environment variables:
    /// `cargo test`'s default in-process thread runner would otherwise race
    /// two such tests against each other (one crate, `#[test]`s not
    /// `nextest`-isolated).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn unique_temp_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!(
            "oxibonsai-testkit-capability-{tag}-{}-{nanos}",
            std::process::id()
        ));
        p
    }

    /// Enforces the invariant that many threads appending to the SAME
    /// manifest path concurrently must never interleave mid-record. Before
    /// the `write_all`-of-one-buffer fix, `writeln!`'s two separate
    /// `write(2)` syscalls made this flaky —
    /// reproduced empirically (30 valid / 3 malformed / 3 blank lines from a
    /// single real test binary's parallel threads). 64 threads x 20 records
    /// each is intentionally far more contention than any real `nextest`
    /// process sees, so a regression here fails reliably rather than
    /// occasionally.
    #[test]
    fn record_at_never_interleaves_under_concurrent_writers() {
        let path = unique_temp_path("concurrent");
        const THREADS: usize = 64;
        const RECORDS_PER_THREAD: usize = 20;

        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let path = &path;
                scope.spawn(move || {
                    for i in 0..RECORDS_PER_THREAD {
                        record_at(
                            path,
                            Capability::Metal,
                            i % 2 == 0,
                            &format!("thread-{t}-record-{i}"),
                            None,
                        );
                    }
                });
            }
        });

        let contents = std::fs::read_to_string(&path).expect("read manifest");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(
            lines.len(),
            THREADS * RECORDS_PER_THREAD,
            "expected exactly {} complete lines, got {} (contents: {contents:?})",
            THREADS * RECORDS_PER_THREAD,
            lines.len()
        );
        for (i, line) in lines.iter().enumerate() {
            let parsed: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("line {i} is not valid JSON: {line:?}: {e}"));
            assert!(
                parsed["capability"].is_string()
                    && parsed["executed"].is_boolean()
                    && parsed["test"].is_string(),
                "line {i} parsed but is missing an expected field: {parsed:?}"
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_appends_one_jsonl_line_per_call() {
        let path = unique_temp_path("append");
        record_at(&path, Capability::Metal, true, "crate::file::test_a", None);
        record_at(&path, Capability::Cuda, false, "crate::file::test_b", None);

        let contents = std::fs::read_to_string(&path).expect("read manifest");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(
            lines.len(),
            2,
            "two record() calls must append exactly two lines, got: {contents:?}"
        );

        let first: serde_json::Value = serde_json::from_str(lines[0]).expect("line 1 valid json");
        assert_eq!(first["capability"], "metal");
        assert_eq!(first["executed"], true);
        assert_eq!(first["test"], "crate::file::test_a");

        let second: serde_json::Value = serde_json::from_str(lines[1]).expect("line 2 valid json");
        assert_eq!(second["capability"], "cuda");
        assert_eq!(second["executed"], false);
        assert_eq!(second["test"], "crate::file::test_b");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn duration_ms_is_omitted_when_not_given_and_present_when_given() {
        let path = unique_temp_path("duration-ms");
        record_at(
            &path,
            Capability::Metal,
            true,
            "crate::file::no_duration",
            None,
        );
        record_at(
            &path,
            Capability::Metal,
            true,
            "crate::file::with_duration",
            Some(1234),
        );

        let contents = std::fs::read_to_string(&path).expect("read manifest");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        // `None` must OMIT the field entirely (not serialize it as `null`),
        // so a record written without a duration stays byte-identical to
        // the pre-`duration_ms` schema.
        assert!(
            !lines[0].contains("duration_ms"),
            "a record with no duration must not mention duration_ms at all: {:?}",
            lines[0]
        );
        let first: serde_json::Value = serde_json::from_str(lines[0]).expect("line 1 valid json");
        assert!(first.get("duration_ms").is_none());

        let second: serde_json::Value = serde_json::from_str(lines[1]).expect("line 2 valid json");
        assert_eq!(second["duration_ms"], 1234);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_executed_timed_writes_a_millisecond_duration() {
        let path = unique_temp_path("record-executed-timed");
        // `report_path()` cannot be redirected from inside this test (it is
        // an env-var-driven free function, not parametrised), so this
        // exercises `record_timed` — the same code `record_executed_timed`
        // calls with `executed = true` — directly against a scratch path,
        // matching this file's other `record_at`-level tests.
        record_at(
            &path,
            Capability::Metal,
            true,
            "crate::file::timed",
            Some(std::time::Duration::from_millis(42).as_millis() as u64),
        );
        let contents = std::fs::read_to_string(&path).expect("read manifest");
        let record: serde_json::Value =
            serde_json::from_str(contents.lines().next().expect("one line")).expect("valid json");
        assert_eq!(record["duration_ms"], 42);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_creates_missing_parent_directories() {
        let mut path = unique_temp_path("nested-dir");
        path.push("a");
        path.push("b");
        path.push("capability-report.json");
        record_at(&path, Capability::ImageParity, true, "t", None);
        assert!(path.exists(), "record_at must create {path:?}'s parents");
        // Clean up the whole unique subtree, not just the leaf file.
        let root = path
            .ancestors()
            .nth(2)
            .expect("path has at least two ancestors")
            .to_path_buf();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn report_path_prefers_oxibonsai_capability_report_env_var() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let expected = unique_temp_path("env-priority");
        std::env::set_var("OXIBONSAI_CAPABILITY_REPORT", &expected);
        std::env::remove_var("CARGO_TARGET_DIR");
        let resolved = report_path();
        std::env::remove_var("OXIBONSAI_CAPABILITY_REPORT");
        assert_eq!(resolved, expected);
    }

    #[test]
    fn report_path_falls_back_to_cargo_target_dir() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("OXIBONSAI_CAPABILITY_REPORT");
        let dir = unique_temp_path("target-dir-fallback");
        std::env::set_var("CARGO_TARGET_DIR", &dir);
        let resolved = report_path();
        std::env::remove_var("CARGO_TARGET_DIR");
        assert_eq!(resolved, dir.join("capability-report.json"));
    }

    #[test]
    fn report_path_falls_back_to_manifest_relative_path_when_no_env_is_set() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("OXIBONSAI_CAPABILITY_REPORT");
        std::env::remove_var("CARGO_TARGET_DIR");
        let resolved = report_path();
        // `Path::starts_with`/`ends_with` compare components literally
        // without normalizing `..`, so the exact expected construction is
        // the clearest correct assertion here (this also pins the
        // "two directories below the workspace root" contract other
        // producers rely on).
        let expected = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("target")
            .join("capability-report.json");
        assert_eq!(resolved, expected);
        assert!(resolved.ends_with("target/capability-report.json"));
    }

    /// The [`record_executed`]/[`record_skipped`] pair must be exactly
    /// [`record`] with the flag spelled out — neither changes the on-disk
    /// manifest format.
    #[test]
    fn record_executed_and_record_skipped_write_the_documented_flag() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let path = unique_temp_path("executed-skipped");
        std::env::set_var("OXIBONSAI_CAPABILITY_REPORT", &path);
        record_executed(Capability::LegacyModels, "crate::file::ran");
        record_skipped(Capability::LegacyModels, "crate::file::skipped");
        std::env::remove_var("OXIBONSAI_CAPABILITY_REPORT");

        let contents = std::fs::read_to_string(&path).expect("read manifest");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "expected two records, got: {contents:?}");

        let ran: serde_json::Value = serde_json::from_str(lines[0]).expect("line 1 valid json");
        assert_eq!(ran["capability"], "legacy-models");
        assert_eq!(ran["executed"], true);
        assert_eq!(ran["test"], "crate::file::ran");

        let skipped: serde_json::Value = serde_json::from_str(lines[1]).expect("line 2 valid json");
        assert_eq!(skipped["capability"], "legacy-models");
        assert_eq!(skipped["executed"], false);
        assert_eq!(skipped["test"], "crate::file::skipped");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn capability_as_str_matches_the_release_gate_contract() {
        assert_eq!(Capability::Metal.as_str(), "metal");
        assert_eq!(Capability::Cuda.as_str(), "cuda");
        assert_eq!(
            Capability::RagRealGeneration.as_str(),
            "rag-real-generation"
        );
        assert_eq!(Capability::ImageParity.as_str(), "image-parity");
        assert_eq!(Capability::LegacyModels.as_str(), "legacy-models");
        assert_eq!(Capability::Bonsai2Models.as_str(), "bonsai2-models");
        // Display must agree with as_str (call sites use both).
        assert_eq!(Capability::Metal.to_string(), Capability::Metal.as_str());
        assert_eq!(
            Capability::Bonsai2Models.to_string(),
            Capability::Bonsai2Models.as_str()
        );
    }

    /// `Capability::Bonsai2Models` must write the
    /// same documented JSONL schema [`record_executed`]/[`record_skipped`]
    /// write for every other capability — this is the dedicated test for the
    /// new variant (distinct from
    /// [`record_executed_and_record_skipped_write_the_documented_flag`],
    /// which pins the shared behaviour via [`Capability::LegacyModels`]).
    #[test]
    fn bonsai2_models_record_executed_and_skipped_write_the_documented_schema() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let path = unique_temp_path("bonsai2-models");
        std::env::set_var("OXIBONSAI_CAPABILITY_REPORT", &path);
        record_executed(Capability::Bonsai2Models, "crate::file::bonsai2_ran");
        record_skipped(Capability::Bonsai2Models, "crate::file::bonsai2_skipped");
        std::env::remove_var("OXIBONSAI_CAPABILITY_REPORT");

        let contents = std::fs::read_to_string(&path).expect("read manifest");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "expected two records, got: {contents:?}");

        let ran: serde_json::Value = serde_json::from_str(lines[0]).expect("line 1 valid json");
        assert_eq!(ran["capability"], "bonsai2-models");
        assert_eq!(ran["executed"], true);
        assert_eq!(ran["test"], "crate::file::bonsai2_ran");

        let skipped: serde_json::Value = serde_json::from_str(lines[1]).expect("line 2 valid json");
        assert_eq!(skipped["capability"], "bonsai2-models");
        assert_eq!(skipped["executed"], false);
        assert_eq!(skipped["test"], "crate::file::bonsai2_skipped");

        let _ = std::fs::remove_file(&path);
    }
}
