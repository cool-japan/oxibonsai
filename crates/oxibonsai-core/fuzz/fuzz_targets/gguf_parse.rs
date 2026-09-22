//! T-03 (second half): a `cargo-fuzz` target for `GgufFile::parse`.
//!
//! `GgufFile::parse` is this crate's untrusted-input boundary — it is the
//! first thing that ever touches bytes read from a `.gguf` file on disk, so
//! it must never panic, overflow, or read/allocate out of bounds on ANY
//! byte sequence, valid or not. Rejecting a malformed file with `Err(..)` is
//! always the correct outcome; panicking or misbehaving on one never is.
//!
//! `crates/oxibonsai-core/tests/fuzz_gguf.rs` already covers this
//! extensively with `proptest` (structured, spec-aware generators) plus a
//! set of permanent regression tests ported from an earlier exploratory
//! fuzzing pass (`crafted_*`, fuzz_gguf.rs:621-737 — see that file's own
//! header comment). Neither is a substitute for this target: `proptest`'s
//! generators are *structured* (they build a plausible-looking GGUF and
//! mutate specific fields), so they cannot discover a crash that depends on
//! byte patterns no generator author anticipated. `cargo fuzz run gguf_parse`
//! explores raw, unconstrained byte sequences with coverage-guided mutation,
//! which is the complementary, unstructured half of the same invariant.
//!
//! Run with (requires nightly and `cargo install cargo-fuzz`):
//! ```text
//! cargo +nightly fuzz run gguf_parse
//! ```
//! from this directory's parent (`crates/oxibonsai-core`), or:
//! ```text
//! cargo +nightly fuzz run --fuzz-dir crates/oxibonsai-core/fuzz gguf_parse
//! ```
//! from the workspace root. Seed corpus: `corpus/gguf_parse/` (a handful of
//! the same crafted byte patterns `fuzz_gguf.rs`'s permanent regression
//! tests assert against, giving libFuzzer's coverage-guided mutation a
//! running start instead of discovering GGUF's magic/header shape from an
//! empty corpus).

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxibonsai_core::gguf::reader::GgufFile;

fuzz_target!(|data: &[u8]| {
    let _ = GgufFile::parse(data);
});
