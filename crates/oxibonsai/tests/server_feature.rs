//! Regression test for the `oxibonsai` facade's `server` (and therefore
//! `full`) feature.
//!
//! `crates/oxibonsai/Cargo.toml` declares `oxibonsai-serve` as an optional
//! dependency, and `src/lib.rs` re-exports it as `oxibonsai::serve` behind
//! `#[cfg(feature = "server")]`. The `server` feature must actually enable
//! `dep:oxibonsai-serve` (not just `oxibonsai-runtime/server`), otherwise the
//! re-export references an extern crate that was never compiled in and the
//! facade fails to build with `--features server` (or `--features full`,
//! which enables `server`).
//!
//! This test only exists in the compiled artifact when the `server` feature
//! is active, so it is a no-op unless `cargo test -p oxibonsai --features
//! server` (or `full`, or `--all-features`) is run — but its mere presence
//! and successful compilation proves the feature wiring is correct.

#![cfg(feature = "server")]

#[test]
fn serve_module_is_reachable_via_server_feature() {
    // Referencing a concrete type from the re-exported `oxibonsai_serve`
    // crate proves that enabling the `server` feature on the `oxibonsai`
    // facade actually activates the optional `oxibonsai-serve` dependency.
    fn assert_type_exists<T>() {}
    assert_type_exists::<oxibonsai::serve::ServerArgs>();
}
