//! Feature flag compile-time validation tests for OxiBonsai.
//!
//! These tests verify that feature flags are correctly propagated across
//! the workspace crate hierarchy: oxibonsai -> oxibonsai-runtime -> oxibonsai-kernels.

// ── Default features ────────────────────────────────────────────────────────

/// The package's `default` feature list names `server`. Read from the manifest
/// rather than from this build's own `cfg`, so the check says the same thing
/// whichever feature set the tests were compiled with (a `cfg`-based check
/// either fails by construction under `--no-default-features` or vanishes).
#[test]
fn default_features_include_server() {
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest = std::fs::read_to_string(&manifest_path).expect("read the package manifest");
    let features = manifest
        .split("\n[features]")
        .nth(1)
        .expect("the manifest has a [features] table");
    let default_line = features
        .lines()
        .take_while(|line| !line.starts_with('['))
        .find(|line| line.trim_start().starts_with("default"))
        .expect("the [features] table has a default list");
    assert!(
        default_line.contains("\"server\""),
        "the `server` feature must be enabled by default; got: {default_line}"
    );
}

// ── Core crate availability ─────────────────────────────────────────────────

#[test]
fn core_crate_exports_available() {
    // Verify that fundamental types from oxibonsai-core are accessible
    let _header_size = std::mem::size_of::<oxibonsai_core::GgufHeader>();
    assert!(_header_size > 0, "GgufHeader should be a non-ZST type");
}

#[test]
fn core_tensor_types_available() {
    let _block_size = std::mem::size_of::<oxibonsai_core::BlockQ1_0G128>();
    // Q1_0_g128: 2-byte FP16 scale + 16 bytes sign bits = 18 bytes
    assert_eq!(_block_size, 18, "BlockQ1_0G128 should be 18 bytes");
}

#[test]
fn core_config_type_available() {
    let _config_size = std::mem::size_of::<oxibonsai_core::Qwen3Config>();
    assert!(_config_size > 0, "Qwen3Config should be a non-ZST type");
}

// ── Kernel crate availability ───────────────────────────────────────────────

#[test]
fn kernel_dispatcher_available() {
    use oxibonsai_kernels::OneBitKernel;
    let dispatcher = oxibonsai_kernels::KernelDispatcher::auto_detect();
    let name = dispatcher.name();
    assert!(
        !name.is_empty(),
        "dispatcher should report a non-empty name"
    );
}

#[test]
fn kernel_trait_available() {
    // Verify the OneBitKernel trait is importable
    fn _assert_trait_exists<T: oxibonsai_kernels::OneBitKernel>() {}
}

// ── Runtime crate availability ──────────────────────────────────────────────

#[test]
fn runtime_sampling_params_available() {
    let params = oxibonsai_runtime::sampling::SamplingParams::default();
    assert!(
        params.temperature >= 0.0,
        "default temperature should be non-negative"
    );
}

#[test]
fn runtime_config_available() {
    // T-09: used to bind `_model`/`_sampling` and assert nothing about
    // either. Pin the actual documented defaults instead (mirrors
    // `runtime_sampling_params_available`'s style just above).
    let config = oxibonsai_runtime::OxiBonsaiConfig::default();
    assert!(
        config.sampling.temperature >= 0.0,
        "default sampling temperature should be non-negative"
    );
    assert!(
        config.model.max_seq_len > 0,
        "default model config should have a positive max sequence length"
    );
}

#[test]
fn runtime_builder_pattern_available() {
    // Verify the builder API is accessible
    let _builder_size = std::mem::size_of::<oxibonsai_runtime::ConfigBuilder>();
    assert!(_builder_size > 0, "ConfigBuilder should be a non-ZST type");
}

#[test]
fn runtime_presets_available() {
    let _preset_size = std::mem::size_of::<oxibonsai_runtime::SamplingPreset>();
    assert!(_preset_size > 0, "SamplingPreset should be a non-ZST type");
}

#[test]
fn runtime_health_available() {
    let _report_size = std::mem::size_of::<oxibonsai_runtime::HealthReport>();
    assert!(_report_size > 0, "HealthReport should be a non-ZST type");
}

#[test]
fn runtime_circuit_breaker_available() {
    let _cb_size = std::mem::size_of::<oxibonsai_runtime::CircuitBreaker>();
    assert!(_cb_size > 0, "CircuitBreaker should be a non-ZST type");
}

#[test]
fn runtime_metrics_available() {
    let metrics = oxibonsai_runtime::InferenceMetrics::new();
    // Verify metrics can render to Prometheus format (starts empty)
    let output = metrics.render_prometheus();
    assert!(
        !output.is_empty(),
        "new metrics should render non-empty Prometheus output"
    );
}

// ── SIMD feature detection ──────────────────────────────────────────────────

// T-09: `simd_feature_flags_compile` used to be three `#[cfg(...)]`-gated
// `assert!(true, ...)` blocks — vacuously true by construction (reaching a
// `#[cfg]`-included block at all already proves it compiled; the `assert!`
// checked nothing an unconditional `{}` would not have).
//
// Re-verified (this differs from the finder's "0 references" evidence,
// which is now stale — a later wave's `src/cli/model_desc.rs` added a real
// `cfg!(feature = "simd-avx2"/"simd-avx512"/"simd-neon")` consumer that
// reports these flags to `oxibonsai info`): the three Cargo features still
// gate **no kernel-dispatch behaviour** anywhere — `KernelTier::Avx2` /
// `::Avx512` / `::Neon` (`crates/oxibonsai-kernels/src/dispatch.rs`) are
// selected purely by `target_arch` plus *runtime* CPU-feature detection
// (`is_x86_feature_detected!`), never by these Cargo features, so there is
// still no dispatch-level behaviour for this test to assert against.
// `model_desc.rs`'s new reporting fields are the only consumer, live in
// `src/cli/` (not reachable from this
// integration-test binary — the root `oxibonsai-cli` package has no `[lib]`
// target `tests/*.rs` files could import from). Deleted per the spec's
// explicit "give it a real assertion or delete it" rather than leaving a
// stale claim or inventing an assertion this file cannot actually make;
// recorded as a deviation for a package that owns `src/cli/model_desc.rs`
// to add real coverage of its own `cfg!()` reporting.

// ── Model crate availability ────────────────────────────────────────────────

#[test]
fn model_types_available() {
    let _model_size = std::mem::size_of::<oxibonsai_model::ModelVariant>();
    assert!(_model_size > 0, "ModelVariant should be a non-ZST type");
}

#[test]
fn model_kv_cache_available() {
    let _cache_size = std::mem::size_of::<oxibonsai_model::KvCache>();
    assert!(_cache_size > 0, "KvCache should be a non-ZST type");
}

// ── Server feature gating ───────────────────────────────────────────────────

// MINOR fix: this used to bind `create_router_with_metrics`
// as compile-time evidence and then `assert!(true, ...)` — no runtime check
// at all beyond "this compiled" (which the `#[cfg(feature = "server")]` gate
// already guarantees). Now it actually calls the function and drives one
// request through the built router, mirroring the in-process
// `tower::ServiceExt::oneshot` style `tests/cli_surface_tests.rs` /
// `tests/rag_serve_cli_tests.rs` already use in this same package.
#[cfg(feature = "server")]
#[tokio::test]
async fn server_module_available_when_feature_enabled() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt as _;

    let engine = oxibonsai_runtime::engine::InferenceEngine::new(
        oxibonsai_core::Qwen3Config::tiny_test(),
        oxibonsai_runtime::sampling::SamplingParams::default(),
        42,
    );
    let router = oxibonsai_runtime::server::create_router_with_metrics(
        engine,
        None,
        std::sync::Arc::new(oxibonsai_runtime::InferenceMetrics::new()),
    );
    let request = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .expect("build request");
    let response = router
        .oneshot(request)
        .await
        .expect("router must handle the request without erroring");
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "a router built by create_router_with_metrics must actually serve GET /health"
    );
}
