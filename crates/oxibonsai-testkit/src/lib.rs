//! `oxibonsai-testkit` — shared, dev-only test-support crate (T-07 / T-05).
//!
//! Never published (`publish = false` in this crate's `Cargo.toml`) and
//! never a runtime dependency of any shipped crate or binary — only ever a
//! `[dev-dependencies]` entry. It exists to stop each test crate from
//! re-inventing the same four things:
//!
//! - [`gguf_fixture`] — one deterministic GGUF fixture builder covering
//!   every quant format this workspace executes, replacing 13 independently
//!   written, subtly-divergent builders (T-07).
//! - [`qwen35_fixture`] — a complete, public synthetic Bonsai 2 hybrid
//!   (`qwen35`) GGUF built on [`gguf_fixture`], for any crate that needs a
//!   loadable hybrid model without a multi-GB real one (EMBED-WIRE handover
//!   (3)).
//! - [`capability`] — the JSONL hardware/fixture-capability self-skip
//!   report contract (T-05), so a skipped hardware-dependent test is
//!   visibly distinct from one that ran and passed.
//! - [`temp_path`] — collision-free temp file helpers built on
//!   `std::env::temp_dir()` (never a hardcoded absolute path).
//! - [`golden`] — golden-vector comparison helpers (`max_abs_diff`,
//!   `cosine_similarity`, `assert_allclose`) for parity tests.
//! - [`workspace`] — resolving this workspace's root and its (gitignored,
//!   often-absent-in-a-worktree) `models/` directory robustly.
//!
//! See this crate's `Cargo.toml` doc comment for why it is currently its
//! own one-crate Cargo workspace rather than a registered member of the
//! main one, and this module's doc comments for the deviation that follows
//! from that (existing duplicate builders could not be re-pointed here yet).

pub mod capability;
pub mod gguf_fixture;
pub mod qwen35_fixture;

/// Collision-free temp-path helpers built on `std::env::temp_dir()`.
///
/// Every helper here resolves through `std::env::temp_dir()` — never a
/// hardcoded absolute path — per this workspace's testing policy.
pub mod temp_path {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A process- and call-unique path under the OS temp directory, with the
    /// given filename `prefix` and `suffix` (e.g. `".gguf"`). Does not
    /// create the file; the caller writes to it (or see
    /// [`write_temp_file`]/[`TempFile`]).
    ///
    /// Uniqueness: process id + a monotonically increasing atomic counter +
    /// wall-clock nanoseconds, so concurrent tests in the same process (or
    /// concurrent `nextest` processes on the same machine) never collide.
    #[must_use]
    pub fn unique_path(prefix: &str, suffix: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxibonsai-testkit-{prefix}-{}-{n}-{nanos}{suffix}",
            std::process::id()
        ));
        path
    }

    /// Write `bytes` to a fresh unique temp file and return its path.
    ///
    /// The caller owns cleanup (the OS temp directory reaps stale files
    /// eventually; use [`TempFile`] for RAII cleanup within one test).
    ///
    /// # Errors
    /// Propagates the underlying `std::fs::write` error.
    pub fn write_temp_file(prefix: &str, suffix: &str, bytes: &[u8]) -> std::io::Result<PathBuf> {
        let path = unique_path(prefix, suffix);
        std::fs::write(&path, bytes)?;
        Ok(path)
    }

    /// An owned temp file that deletes itself on drop.
    #[derive(Debug)]
    pub struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        /// Write `bytes` to a fresh unique temp file and wrap it for RAII
        /// cleanup.
        ///
        /// # Errors
        /// Propagates the underlying `std::fs::write` error.
        pub fn write(prefix: &str, suffix: &str, bytes: &[u8]) -> std::io::Result<Self> {
            Ok(Self {
                path: write_temp_file(prefix, suffix, bytes)?,
            })
        }

        /// The file's path.
        #[must_use]
        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn unique_path_never_collides_across_many_calls() {
            let mut seen = std::collections::HashSet::new();
            for _ in 0..256 {
                let p = unique_path("collision-check", ".bin");
                assert!(seen.insert(p), "unique_path produced a duplicate");
            }
        }

        #[test]
        fn write_temp_file_roundtrips_bytes() {
            let path = write_temp_file("roundtrip", ".gguf", b"GGUF-test-bytes").expect("write");
            let back = std::fs::read(&path).expect("read back");
            assert_eq!(back, b"GGUF-test-bytes");
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn temp_file_deletes_itself_on_drop() {
            let path = {
                let f = TempFile::write("raii", ".gguf", b"data").expect("write");
                let p = f.path().to_path_buf();
                assert!(p.exists());
                p
            };
            assert!(!path.exists(), "TempFile must delete its file on drop");
        }

        #[test]
        fn paths_are_never_hardcoded_absolute_literals() {
            // Regression guard for the project rule "never hardcode
            // absolute paths": every path this module returns must live
            // under std::env::temp_dir(), not some fixed literal.
            let p = unique_path("policy-check", ".tmp");
            assert!(p.starts_with(std::env::temp_dir()));
        }
    }
}

/// Golden-vector comparison helpers shared by every parity/regression test.
pub mod golden {
    /// The largest absolute difference between `actual` and `expected`.
    ///
    /// # Panics
    /// Panics if the two slices have different lengths — this is a test
    /// helper, and a length mismatch is always a test bug, never expected
    /// input worth a `Result`.
    #[must_use]
    pub fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(
            actual.len(),
            expected.len(),
            "max_abs_diff: length mismatch ({} vs {})",
            actual.len(),
            expected.len()
        );
        actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max)
    }

    /// Cosine similarity between two equal-length vectors, in `[-1.0, 1.0]`
    /// (`1.0` for identical direction). Returns `0.0` if either vector is
    /// all-zero, rather than the `NaN` a raw division would give.
    ///
    /// # Panics
    /// Panics on a length mismatch (see [`max_abs_diff`]).
    #[must_use]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let norm_a: f64 = a
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        let norm_b: f64 = b
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)) as f32
    }

    /// Asserts every element of `actual` is within `atol + rtol *
    /// |expected|` of `expected` (numpy's `allclose` convention), panicking
    /// with the first offending index and both values on failure.
    ///
    /// # Panics
    /// On a length mismatch, or the first out-of-tolerance element.
    pub fn assert_allclose(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "assert_allclose: length mismatch ({} vs {})",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let bound = atol + rtol * e.abs();
            let diff = (a - e).abs();
            assert!(
                diff <= bound,
                "assert_allclose: index {i}: actual={a} expected={e} diff={diff} > bound={bound} (atol={atol}, rtol={rtol})"
            );
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn max_abs_diff_finds_the_largest_gap() {
            let a = [1.0, 2.0, 3.0];
            let b = [1.0, 2.5, 2.0];
            assert!((max_abs_diff(&a, &b) - 1.0).abs() < 1e-6);
        }

        #[test]
        #[should_panic(expected = "length mismatch")]
        fn max_abs_diff_panics_on_length_mismatch() {
            let _ = max_abs_diff(&[1.0], &[1.0, 2.0]);
        }

        #[test]
        fn cosine_similarity_is_one_for_identical_vectors() {
            let a = [1.0, 2.0, 3.0, -4.0];
            assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_is_minus_one_for_opposite_vectors() {
            let a = [1.0, 2.0, 3.0];
            let b = [-1.0, -2.0, -3.0];
            assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_is_zero_for_orthogonal_vectors() {
            let a = [1.0, 0.0];
            let b = [0.0, 1.0];
            assert!(cosine_similarity(&a, &b).abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_of_zero_vector_is_zero_not_nan() {
            let a = [0.0, 0.0, 0.0];
            let b = [1.0, 2.0, 3.0];
            assert_eq!(cosine_similarity(&a, &b), 0.0);
        }

        #[test]
        fn assert_allclose_accepts_within_tolerance() {
            assert_allclose(&[1.0001], &[1.0], 1e-3, 0.0);
        }

        #[test]
        #[should_panic(expected = "diff=")]
        fn assert_allclose_rejects_outside_tolerance() {
            assert_allclose(&[1.1], &[1.0], 1e-3, 0.0);
        }
    }
}

/// Locating this workspace's root and its `models/` directory.
///
/// (Wave-1 gatekeeper OPTIONAL #O7): several real-model acceptance tests
/// (`crates/oxibonsai-core/tests/quant_prism_golden.rs`,
/// `crates/oxibonsai-model/src/gguf_loader.rs`'s own test module — neither
/// owned by this package) silently no-op when `models/` is empty, which is
/// the normal state inside an isolated wave worktree (confirmed empirically
/// this wave: this worktree's own `models/` holds only a `.gitkeep`). This
/// module is the fix's building block: it resolves the real `models/`
/// directory the same, robust way regardless of which crate's test binary
/// calls in, with an env-var override for CI or a custom layout, and it is
/// what those two files should call once they can take this crate as a
/// dev-dependency (see the `Cargo.toml` doc comment — that wiring is a
/// deviation, this helper is not). It deliberately does not hardcode, guess
/// at, or symlink to any *other* checkout's path (e.g. the main tree this
/// worktree was created from): only `$OXIBONSAI_MODELS_DIR`, set by
/// whatever created the environment, can point here at anything outside
/// this workspace's own `models/` directory.
pub mod workspace {
    use std::path::PathBuf;

    /// The workspace root, resolved from this crate's own (fixed at compile
    /// time) `CARGO_MANIFEST_DIR`: every workspace member lives exactly two
    /// directories below the root as `crates/<name>`, so this is stable
    /// regardless of which crate's test binary actually calls in — the same
    /// technique [`crate::capability::report_path`] uses for the workspace
    /// `target/` directory.
    #[must_use]
    pub fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    /// The real-model directory: `$OXIBONSAI_MODELS_DIR` if set to a
    /// non-empty value, else `<workspace root>/models`.
    #[must_use]
    pub fn models_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("OXIBONSAI_MODELS_DIR") {
            if !dir.is_empty() {
                return PathBuf::from(dir);
            }
        }
        root().join("models")
    }

    /// `models_dir().join(filename)`, if that path exists and is a
    /// non-empty file — `None` otherwise (including the common "gitignored,
    /// absent in a fresh worktree" case), so callers can self-skip on
    /// `None` rather than opening a zero-byte placeholder.
    #[must_use]
    pub fn find_model(filename: &str) -> Option<PathBuf> {
        let path = models_dir().join(filename);
        match std::fs::metadata(&path) {
            Ok(meta) if meta.is_file() && meta.len() > 0 => Some(path),
            _ => None,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::Mutex;

        /// Guards this module's two env-mutating tests against each other
        /// (independent of `capability`'s own lock: different env vars).
        static ENV_LOCK: Mutex<()> = Mutex::new(());

        #[test]
        fn root_contains_this_crates_own_cargo_toml_two_levels_up() {
            let candidate = root().join("crates/oxibonsai-testkit/Cargo.toml");
            assert!(
                candidate.exists(),
                "root() = {:?} does not contain this crate at {:?}",
                root(),
                candidate
            );
        }

        #[test]
        fn models_dir_env_override_takes_priority() {
            let _guard = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // MINOR (verifier wave 3): this path is never opened (only
            // compared against `models_dir()`'s return value), but a fixed
            // absolute literal would still trip a naive "never hardcode
            // absolute paths" policy grep. `std::env::temp_dir()` satisfies
            // the policy and the override-priority assertion equally well.
            let override_path = std::env::temp_dir().join("oxibonsai-testkit-override-check");
            std::env::set_var("OXIBONSAI_MODELS_DIR", &override_path);
            assert_eq!(models_dir(), override_path);
            std::env::remove_var("OXIBONSAI_MODELS_DIR");
        }

        #[test]
        fn find_model_returns_none_for_a_missing_file() {
            let _guard = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::env::remove_var("OXIBONSAI_MODELS_DIR");
            assert!(find_model("this-file-does-not-exist-in-any-checkout.gguf").is_none());
        }

        #[test]
        fn find_model_finds_this_crates_own_cargo_toml_as_a_smoke_test() {
            let _guard = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::env::set_var(
                "OXIBONSAI_MODELS_DIR",
                root().join("crates/oxibonsai-testkit"),
            );
            assert!(
                find_model("Cargo.toml").is_some(),
                "find_model must locate an existing non-empty file"
            );
            std::env::remove_var("OXIBONSAI_MODELS_DIR");
        }
    }
}
