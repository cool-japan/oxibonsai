//! Unit tests for `cmd_serve.rs`, split into a sibling file to keep
//! `cmd_serve.rs` under the 2000-line policy limit (declared there via
//! `#[path]`, so `super` still names that module).

use super::*;
use crate::cli::util::test_env::{self, EnvVarGuard};

// ── is_loopback_host / bind_safety_check ────────────────────────────────

#[test]
fn loopback_hosts_are_recognized() {
    for host in ["127.0.0.1", "127.5.5.5", "::1", "localhost", "LOCALHOST"] {
        assert!(is_loopback_host(host), "{host} should be loopback");
    }
}

#[test]
fn non_loopback_hosts_are_not_loopback() {
    for host in ["0.0.0.0", "192.168.1.1", "example.com", ""] {
        assert!(!is_loopback_host(host), "{host} should not be loopback");
    }
}

#[test]
fn loopback_bind_never_needs_auth() {
    bind_safety_check("127.0.0.1", None, false).expect("loopback bind is always safe");
}

#[test]
fn non_loopback_bind_without_auth_is_refused() {
    let err = bind_safety_check("0.0.0.0", None, false).expect_err("must refuse");
    assert!(err.to_string().contains("0.0.0.0"));
}

#[test]
fn non_loopback_bind_with_bearer_token_is_allowed() {
    bind_safety_check("0.0.0.0", Some(&"x".repeat(20)), false)
        .expect("a configured bearer token makes this safe");
}

#[test]
fn non_loopback_bind_with_insecure_no_auth_is_allowed() {
    bind_safety_check("0.0.0.0", None, true).expect("explicit opt-out is honored");
}

// ── resolve_admin_auth_from (FIX2-SERVE item 4: admin-token env asymmetry) ──
//
// Rewritten as a verifier follow-up on this same finding: these used to
// exercise `resolve_admin_auth` directly by mutating the two SAME
// process-global env vars (`OXIBONSAI_ADMIN_TOKEN`, `OXI_ADMIN_TOKEN`)
// via `unsafe { std::env::set_var / remove_var }` with no
// serialization -- flaky against each other under `cargo test`'s
// default parallel test threads, and unsound against `router_tests`'
// concurrent `getenv` on `OXI_ADMIN_TOKEN` (see `resolve_admin_auth_from`'s
// doc comment above for the full explanation). They now call the pure
// `resolve_admin_auth_from` with both the candidate token and the
// pre-resolved legacy policy as plain arguments, so none of them touch
// either environment variable at all -- and each asserts a distinct,
// concrete expected outcome rather than delegating to a fresh
// `AdminAuthConfig::from_env()` call, so a broken fallback arm cannot
// hide behind `OXI_ADMIN_TOKEN` happening to be unset in the test
// environment.

#[test]
fn resolve_admin_auth_from_prefers_new_token_over_legacy() {
    let resolved = resolve_admin_auth_from(
        Some("new-convention-token-value"),
        AdminAuthConfig::with_admin_token("legacy-convention-token-value"),
    );
    assert_eq!(
        resolved,
        AdminAuthConfig::with_admin_token("new-convention-token-value")
    );
}

#[test]
fn resolve_admin_auth_from_falls_back_to_legacy_when_new_token_is_none() {
    let resolved = resolve_admin_auth_from(
        None,
        AdminAuthConfig::with_admin_token("legacy-convention-token-value"),
    );
    assert_eq!(
        resolved,
        AdminAuthConfig::with_admin_token("legacy-convention-token-value")
    );
}

#[test]
fn resolve_admin_auth_from_falls_back_to_legacy_when_new_token_is_empty() {
    let resolved = resolve_admin_auth_from(
        Some(""),
        AdminAuthConfig::with_admin_token("legacy-convention-token-value"),
    );
    assert_eq!(
        resolved,
        AdminAuthConfig::with_admin_token("legacy-convention-token-value")
    );
}

#[test]
fn resolve_admin_auth_from_locked_when_neither_is_configured() {
    let resolved = resolve_admin_auth_from(None, AdminAuthConfig::locked());
    assert_eq!(resolved, AdminAuthConfig::locked());
}

#[test]
fn resolve_admin_auth_from_ignores_legacy_when_new_token_is_whitespace_only() {
    // A previously-untested edge case documented on
    // `resolve_admin_auth_from`'s doc comment: a whitespace-only new
    // token is non-empty (so it does NOT hit the fallback arm), but
    // `AdminAuthConfig::with_admin_token` itself treats whitespace-only
    // as "not configured", so this locks even though a legacy token is
    // available -- unlike an *empty* new token
    // (`resolve_admin_auth_from_falls_back_to_legacy_when_new_token_is_empty`
    // above), which does fall back to it.
    let resolved = resolve_admin_auth_from(
        Some("   "),
        AdminAuthConfig::with_admin_token("legacy-convention-token-value"),
    );
    assert_eq!(resolved, AdminAuthConfig::locked());
}

// ── default_sampling_params (FIX2-SERVE item 6: greedy penalties) ───────

/// THE regression test for the serve half of the P0 CPU-vs-Metal
/// divergence: `oxibonsai serve`'s baseline `SamplingParams` must match
/// what `oxibonsai run`/`chat`/`benchmark`/`eval` resolve to for the
/// same (default, no-flag) inputs -- both now go through the identical
/// shared `build_sampling_params` constructor. `SamplingParams` carries
/// no `PartialEq` (it is owned by `oxibonsai-runtime`, not this
/// package), so fields are compared individually.
///
/// The literals themselves now live in ONE place (`util::DEFAULT_*`, read
/// by `mod.rs`, `cmd_run::resolve_sampling` and `baseline_sampling_params`),
/// so the two paths can no longer drift apart silently; the
/// non-tautological cross-path check — each path running its OWN
/// resolution code over the same parsed GGUF — is
/// [`serve_and_run_resolve_the_same_baseline_for_the_same_gguf`] below. What
/// this test pins: the no-declaration baseline equals the documented
/// literals, and the server's baseline never again silently reverts to the
/// hidden `repetition_penalty` of `1.1` an old `SamplingParams::default()`
/// carried.
#[test]
fn serve_default_sampling_params_match_the_run_path_for_the_same_inputs() {
    let serve_params = default_sampling_params();
    // "The run path, same inputs": the literal defaults `src/cli/mod.rs`
    // resolves `run`'s `--temperature`/`--top-k`/`--top-p`/
    // `--repetition-penalty` flags to when none of them are passed.
    let run_path_params = build_sampling_params(0.7, 40, 0.9, 1.0);

    assert!(
        (serve_params.temperature - run_path_params.temperature).abs() < f32::EPSILON,
        "temperature mismatch: serve={} run={}",
        serve_params.temperature,
        run_path_params.temperature
    );
    assert_eq!(serve_params.top_k, run_path_params.top_k);
    assert!(
        (serve_params.top_p - run_path_params.top_p).abs() < f32::EPSILON,
        "top_p mismatch: serve={} run={}",
        serve_params.top_p,
        run_path_params.top_p
    );
    assert!(
        (serve_params.repetition_penalty - run_path_params.repetition_penalty).abs() < f32::EPSILON,
        "repetition_penalty mismatch: serve={} run={}",
        serve_params.repetition_penalty,
        run_path_params.repetition_penalty
    );
    assert_eq!(serve_params.max_tokens, run_path_params.max_tokens);

    // The concrete regression this guards against: the server's
    // baseline must never again silently carry
    // `SamplingParams::default()`'s hidden `repetition_penalty` of
    // `1.1` -- "greedy means greedy" (a per-request `temperature: 0`)
    // must hold on `oxibonsai serve` exactly as it does on `oxibonsai
    // run`.
    assert!((serve_params.repetition_penalty - 1.0).abs() < f32::EPSILON);
}

/// RT-17 across the two paths: for the same parsed GGUF, `serve`'s pool
/// baseline ([`baseline_sampling_params`]) and what `run`/`chat` resolve
/// with no flag and no `--config` value (`cmd_run::resolve_sampling` +
/// the shared constructor) are the same parameters — for a file that
/// declares nothing (the legacy models) and for one that declares the
/// Bonsai 2 defaults (`temp` 1.0, `top_p` 0.95, `top_k` 20 as INT32).
#[test]
fn serve_and_run_resolve_the_same_baseline_for_the_same_gguf() {
    use oxibonsai_core::gguf::writer::{GgufWriter, MetadataWriteValue};
    let declared = |entries: Vec<(&str, MetadataWriteValue)>| {
        let mut w = GgufWriter::new();
        w.add_metadata(
            "general.architecture",
            MetadataWriteValue::Str("qwen35".to_string()),
        );
        for (key, value) in entries {
            w.add_metadata(key, value);
        }
        w.to_bytes().expect("serialize")
    };
    for bytes in [
        declared(Vec::new()),
        declared(vec![
            ("general.sampling.temp", MetadataWriteValue::F32(1.0)),
            ("general.sampling.top_p", MetadataWriteValue::F32(0.95)),
            ("general.sampling.top_k", MetadataWriteValue::I32(20)),
        ]),
    ] {
        let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&bytes).expect("parse");
        let serve = baseline_sampling_params(
            &oxibonsai_runtime::sampling::GgufSamplingDefaults::from_metadata(&gguf.metadata),
        );
        let run = crate::cli::cmd_run::resolve_sampling(None, None, None, None, &gguf.metadata)
            .expect("valid");
        let run = build_sampling_params(
            run.temperature,
            run.top_k,
            run.top_p,
            crate::cli::util::DEFAULT_REPETITION_PENALTY,
        );
        assert_eq!(serve.temperature.to_bits(), run.temperature.to_bits());
        assert_eq!(serve.top_k, run.top_k);
        assert_eq!(serve.top_p.to_bits(), run.top_p.to_bits());
        assert_eq!(
            serve.repetition_penalty.to_bits(),
            run.repetition_penalty.to_bits()
        );
    }
}

/// The dedicated `/v1/embeddings` engine honours `--backend` and
/// `--rope-scaling` exactly like the pool's replicas: on a `--backend cpu`
/// server it runs a CPU tier (never uploading weights to the GPU), and
/// `--rope-scaling off` builds it with plain RoPE on a YaRN file.
#[test]
fn the_embedding_engine_honours_backend_and_rope_scaling() {
    use oxibonsai_core::gguf::writer::MetadataWriteValue;
    use oxibonsai_runtime::config::RopeScalingMode;
    use oxibonsai_runtime::engine_seam::Backend;
    let bytes = crate::cli::test_fixtures::tiny_dense_gguf(vec![
        (
            "qwen3.rope.scaling.type",
            MetadataWriteValue::Str("yarn".to_string()),
        ),
        ("qwen3.rope.scaling.factor", MetadataWriteValue::F32(4.0)),
        (
            "qwen3.rope.scaling.original_context_length",
            MetadataWriteValue::U32(128),
        ),
    ]);
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static> = Box::leak(Box::new(
        oxibonsai_core::gguf::reader::GgufFile::parse(bytes).expect("parse"),
    ));
    let params = build_sampling_params(0.0, 40, 0.9, 1.0);
    let built = oxibonsai_runtime::engine_pool::build_pool_from_static_gguf_with_rope(
        gguf,
        params.clone(),
        42,
        64,
        Some(1),
        Backend::Cpu,
        RopeScalingMode::Off.into(),
    )
    .expect("pool");
    let load = EmbeddingEngineLoad {
        params,
        seed: 42,
        max_seq_len: 64,
        backend: Backend::Cpu,
        rope_scaling: RopeScalingMode::Off,
    };
    let engine = build_embedding_engine(&built, &load, 32).expect("embedding engine");
    assert!(
        !engine.uses_fused_gpu_decode() && !engine.greedy_gpu_eligible(false),
        "--backend cpu: the embedding engine must not run a GPU tier ({})",
        engine.kernel_label()
    );
    assert_eq!(engine.backend(), Backend::Cpu);
    let rope = engine
        .dense_model()
        .expect("dense fixture")
        .config()
        .rope_scaling
        .clone();
    assert_eq!(
        rope,
        oxibonsai_core::config::RopeScaling::None,
        "--rope-scaling off"
    );

    let auto = EmbeddingEngineLoad {
        rope_scaling: RopeScalingMode::Auto,
        ..load
    };
    let engine = build_embedding_engine(&built, &auto, 32).expect("embedding engine");
    assert!(matches!(
        engine.dense_model().expect("dense").config().rope_scaling,
        oxibonsai_core::config::RopeScaling::Yarn { .. }
    ));
}

// ── resolve_seed ─────────────────────────────────────────────────────────

#[test]
fn resolve_seed_env_override_wins() {
    let _env = test_env::lock();
    let _seed = EnvVarGuard::set("OXIBONSAI_SEED", "123456789");
    assert_eq!(resolve_seed(), 123_456_789);
}

#[test]
fn resolve_seed_ignores_malformed_override() {
    let _env = test_env::lock();
    let _seed = EnvVarGuard::set("OXIBONSAI_SEED", "not-a-number");
    // Falls back to the pseudo-random (time-derived) path; two calls a
    // moment apart must not be a hardcoded constant. The nanosecond
    // component alone makes two calls separated by any measurable delay
    // collide with astronomically low probability, and a constant
    // fallback -- the actual regression this guards against -- would
    // fail it deterministically.
    let first = resolve_seed();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let second = resolve_seed();
    assert_ne!(
        first, second,
        "the pseudo-random fallback must not be a constant"
    );
}

// ── resolve_bearer_token_file_override (SV-33) ──────────────────────────
//
// ENGINE-SEAM addendum (3): every test that touches
// `OXIBONSAI_BEARER_TOKEN_FILE` holds the crate-wide env lock and restores
// the variable through an RAII guard (a panicking assertion still restores
// it). The `--bearer-token-file` flag path is a plain parameter, so its
// tests touch no environment at all.

fn token_file(tag: &str, contents: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = crate::cli::test_fixtures::scratch_dir(&format!("bearer_{tag}"));
    let path = dir.join("token");
    std::fs::write(&path, contents).expect("write token file");
    (dir, path)
}

#[test]
fn bearer_token_file_override_passes_through_when_unset() {
    let _env = test_env::lock();
    let _unset = EnvVarGuard::remove("OXIBONSAI_BEARER_TOKEN_FILE");
    let result = resolve_bearer_token_file_override(None, Some("flag-value".to_string()))
        .expect("unset var must not error");
    assert_eq!(result, Some("flag-value".to_string()));
}

#[test]
fn bearer_token_file_override_wins_and_is_trimmed() {
    let (dir, path) = token_file("env_wins", "  from-file  \n");
    let result = {
        let _env = test_env::lock();
        let _file = EnvVarGuard::set("OXIBONSAI_BEARER_TOKEN_FILE", &path);
        resolve_bearer_token_file_override(None, Some("flag-value".to_string()))
    };
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(result.expect("resolve"), Some("from-file".to_string()));
}

#[test]
fn bearer_token_file_override_errors_on_missing_file() {
    // Never a hardcoded absolute path (project policy): a process-unique
    // path under the real temp dir.
    let missing = std::env::temp_dir().join(format!(
        "oxibonsai_cli_definitely_missing_bearer_token_file_{}",
        std::process::id()
    ));
    let _env = test_env::lock();
    let _file = EnvVarGuard::set("OXIBONSAI_BEARER_TOKEN_FILE", &missing);
    let err = resolve_bearer_token_file_override(None, None).expect_err("missing file");
    assert!(
        err.to_string().contains("OXIBONSAI_BEARER_TOKEN_FILE"),
        "{err}"
    );
}

#[test]
fn bearer_token_file_flag_path_is_read_without_touching_the_environment() {
    let (dir, path) = token_file("flag", "\tflag-file-token\n");
    let path_str = path.to_string_lossy().into_owned();
    let result = resolve_bearer_token_file_override(Some(&path_str), Some("ignored".to_string()));
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        result.expect("resolve"),
        Some("flag-file-token".to_string())
    );
}

#[test]
fn bearer_token_file_flag_path_wins_over_the_environment_variable() {
    let (dir_flag, flag_path) = token_file("flag_over_env", "from-flag");
    let (dir_env, env_path) = token_file("env_loses", "from-env");
    let flag_str = flag_path.to_string_lossy().into_owned();
    let result = {
        let _env = test_env::lock();
        let _file = EnvVarGuard::set("OXIBONSAI_BEARER_TOKEN_FILE", &env_path);
        resolve_bearer_token_file_override(Some(&flag_str), None)
    };
    let _ = std::fs::remove_dir_all(&dir_flag);
    let _ = std::fs::remove_dir_all(&dir_env);
    assert_eq!(result.expect("resolve"), Some("from-flag".to_string()));
}

#[test]
fn bearer_token_file_flag_path_errors_name_the_flag_and_reject_empty_files() {
    let (dir, path) = token_file("empty", "   \n");
    let path_str = path.to_string_lossy().into_owned();
    let err = resolve_bearer_token_file_override(Some(&path_str), None).expect_err("empty");
    let _ = std::fs::remove_dir_all(&dir);
    let msg = err.to_string();
    assert!(
        msg.contains("--bearer-token-file") && msg.contains("empty"),
        "{msg}"
    );
}

// ── checksum helpers ─────────────────────────────────────────────────────
//
// `lookup_expected_checksum` has no non-test call site left in this
// *binary* target now that `verify_model_checksum`'s implementation
// lives in `oxibonsai_runtime::serve_shared` -- imported here, inside
// `#[cfg(test)]`, rather than at module level, so a non-test build of
// this bin target does not see (and cannot warn about) an otherwise
// -unused import.
use oxibonsai_runtime::serve_shared::lookup_expected_checksum;

#[test]
fn lookup_matches_by_bare_filename() {
    let text =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  models/Foo.gguf\n";
    let got = lookup_expected_checksum(text, Path::new("/elsewhere/Foo.gguf"));
    assert_eq!(
        got,
        Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string())
    );
}

#[test]
fn lookup_is_none_for_unknown_file() {
    let text =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  models/Foo.gguf\n";
    assert!(lookup_expected_checksum(text, Path::new("models/Unknown.gguf")).is_none());
}

#[test]
fn verify_is_a_noop_when_checksums_file_is_absent() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "oxibonsai_cli_checksum_test_{}",
        std::process::id()
    ));
    let missing_checksums = tmp_dir.join("does-not-exist.sha256");
    let model = tmp_dir.join("Model.gguf");
    verify_model_checksum(&model, &missing_checksums)
        .expect("missing checksums file must not block loading");
}

// NOTE (sec-12 verifier finding): the previous version of this test
// ("verify_warns_but_does_not_fail_when_hasher_is_unavailable") never
// actually wrote a model file, only the checksums entry -- so it
// exercised `verify_model_checksum`'s `Err(_)` (unreadable file)
// branch, not an "unavailable hasher" branch, regardless of what
// `compute_sha256_hex` returned. It is replaced below by tests that
// exercise the real equality/mismatch branches end to end through this
// binary's own imported `verify_model_checksum`, plus one correctly
// named/scoped test for the unreadable-file case it was accidentally
// covering.

#[test]
fn verify_succeeds_when_the_checksum_really_matches() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "oxibonsai_cli_checksum_test_matches_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
    let model = tmp_dir.join("Model.gguf");
    std::fs::write(&model, b"the real bytes of a model file").expect("write model");
    let hex = oxibonsai_runtime::serve_shared::compute_sha256_hex(&model)
        .expect("compute hex")
        .expect("digest is always Some for a readable file");
    let checksums_path = tmp_dir.join("checksums.sha256");
    std::fs::write(
        &checksums_path,
        format!(
            "{hex}  {}\n",
            model.file_name().expect("filename").to_string_lossy()
        ),
    )
    .expect("write checksums file");

    verify_model_checksum(&model, &checksums_path)
        .expect("a genuinely matching checksum must not block loading");
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

/// THE regression test finding `sec-12` was missing: a known-good
/// checksum entry that no longer matches the file on disk must be
/// **fatal** -- this is `verify_model_checksum`'s one fatal branch, and
/// it was unreachable dead code while `compute_sha256_hex` always
/// returned `Ok(None)`.
#[test]
fn verify_fails_when_a_known_good_entry_no_longer_matches_a_corrupted_file() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "oxibonsai_cli_checksum_test_corrupted_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
    let model = tmp_dir.join("Model.gguf");

    std::fs::write(&model, b"the original, known-good model bytes").expect("write model");
    let good_hex = oxibonsai_runtime::serve_shared::compute_sha256_hex(&model)
        .expect("compute hex")
        .expect("digest is always Some for a readable file");
    let checksums_path = tmp_dir.join("checksums.sha256");
    std::fs::write(
        &checksums_path,
        format!(
            "{good_hex}  {}\n",
            model.file_name().expect("filename").to_string_lossy()
        ),
    )
    .expect("write checksums file");

    // Corrupt the file in place; the checksums file still lists the
    // ORIGINAL (now stale) digest as "known-good".
    std::fs::write(
        &model,
        b"corrupted! these are not the original bytes at all",
    )
    .expect("corrupt model");

    let err = verify_model_checksum(&model, &checksums_path)
        .expect_err("a stale known-good entry against corrupted bytes must be fatal");
    assert!(
        err.contains("checksum mismatch"),
        "error should explain the mismatch: {err}"
    );
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn verify_degrades_gracefully_when_the_model_file_cannot_be_opened() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "oxibonsai_cli_checksum_test_missing_model_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
    let model = tmp_dir.join("Model.gguf"); // Deliberately never created.
    let checksums_path = tmp_dir.join("checksums.sha256");
    std::fs::write(&checksums_path, format!("{}  Model.gguf\n", "c".repeat(64)))
        .expect("write checksums file");
    verify_model_checksum(&model, &checksums_path)
        .expect("an unreadable model file must not block loading (degrades to a warning)");
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// ── env helpers ──────────────────────────────────────────────────────────

#[test]
fn env_cors_origins_splits_and_trims_when_set() {
    let origins = {
        let _env = test_env::lock();
        let _cors = EnvVarGuard::set(
            "OXIBONSAI_CORS_ORIGIN",
            "https://a.example.com, https://b.example.com ,",
        );
        env_cors_origins()
    };
    assert_eq!(
        origins,
        vec![
            "https://a.example.com".to_string(),
            "https://b.example.com".to_string()
        ]
    );
}

#[test]
fn env_flag_accepts_common_spellings() {
    for (raw, expected) in [
        ("1", true),
        ("true", true),
        ("YES", true),
        ("on", true),
        ("0", false),
        ("false", false),
        ("no", false),
        ("garbage", false),
    ] {
        let got = {
            let _env = test_env::lock();
            let _flag = EnvVarGuard::set("OXIBONSAI_TEST_FLAG_CMD_SERVE", raw);
            env_flag("OXIBONSAI_TEST_FLAG_CMD_SERVE")
        };
        assert_eq!(got, expected, "raw={raw:?}");
    }
}

// ── --embedding-backend tfidf: corpus validation ────────────────────────────

#[test]
fn tfidf_needs_a_corpus_and_a_corpus_needs_tfidf() {
    let err = resolve_tfidf_corpus(EmbeddingBackendChoice::Tfidf, None).expect_err("no corpus");
    assert!(err.to_string().contains("--embedding-corpus"), "{err}");
    let err = resolve_tfidf_corpus(EmbeddingBackendChoice::Model, Some("c.txt"))
        .expect_err("corpus without tfidf");
    assert!(
        err.to_string()
            .contains("only applies with --embedding-backend tfidf"),
        "{err}"
    );
    assert!(resolve_tfidf_corpus(EmbeddingBackendChoice::None, None)
        .expect("fine")
        .is_none());
}

#[test]
fn the_corpus_is_one_trimmed_document_per_non_blank_line() {
    let dir = crate::cli::test_fixtures::scratch_dir("tfidf_corpus");
    let path = dir.join("corpus.txt");
    std::fs::write(&path, "  first doc \n\n\tsecond doc\n   \n").expect("write");
    let docs = resolve_tfidf_corpus(
        EmbeddingBackendChoice::Tfidf,
        Some(path.to_string_lossy().as_ref()),
    )
    .expect("readable")
    .expect("tfidf");
    assert_eq!(
        docs,
        vec!["first doc".to_string(), "second doc".to_string()]
    );

    let empty = dir.join("empty.txt");
    std::fs::write(&empty, "\n  \n").expect("write");
    let err = load_embedding_corpus(empty.to_string_lossy().as_ref()).expect_err("no documents");
    assert!(err.to_string().contains("holds no documents"), "{err}");
    let err = load_embedding_corpus(dir.join("missing.txt").to_string_lossy().as_ref())
        .expect_err("missing");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        err.to_string()
            .contains("failed to read --embedding-corpus"),
        "{err}"
    );
}

// ── harden_router: full composition against a tiny in-memory pool ──────

mod router_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use oxibonsai_core::config::Qwen3Config;
    use oxibonsai_runtime::engine::InferenceEngine;
    use oxibonsai_runtime::engine_pool::EnginePool;
    use oxibonsai_runtime::metrics::InferenceMetrics;
    use oxibonsai_runtime::sampling::SamplingParams;
    use std::time::Duration;
    use tower::ServiceExt;

    fn default_opts() -> HardeningOptions {
        HardeningOptions {
            bearer_token: None,
            max_concurrent_requests: 32,
            request_timeout_ms: 60_000,
            max_body_bytes: 4 * 1024 * 1024,
            cors_origins: Vec::new(),
            cors_allow_credentials: false,
            rate_limit_rpm: None,
            rate_limit_burst: 20.0,
            chat_defaults: ChatDefaults::default(),
            embeddings_override: None,
        }
    }

    // ── --embedding-backend tfidf ──────────────────────────────────────────

    async fn post_embeddings(router: &Router, body: &str) -> (StatusCode, serde_json::Value) {
        let resp = router
            .clone()
            .oneshot(
                Request::post("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .expect("request"),
            )
            .await
            .expect("response");
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body");
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    /// Without an override the runtime router's model-only route answers
    /// the honest 501; with `--embedding-backend tfidf` the same request is
    /// answered from the vocabulary fitted on the corpus — deterministic,
    /// reported as `"tfidf"`, and still behind the bearer layer.
    #[tokio::test]
    async fn tfidf_embeddings_are_served_from_the_fitted_corpus() {
        let body = r#"{"input":["the cat sat on the mat","a dog ran"],"model":"m"}"#;
        let plain = router_for(default_opts());
        let (status, _) = post_embeddings(&plain, body).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);

        let corpus: Vec<String> = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "a bird sang on the tree",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut opts = default_opts();
        opts.embeddings_override = Some(build_tfidf_embeddings_router(
            &corpus,
            Arc::new(InferenceMetrics::new()),
        ));
        let router = router_for(opts);
        let (status, first) = post_embeddings(&router, body).await;
        assert_eq!(status, StatusCode::OK, "{first}");
        assert_eq!(
            first["model"],
            serde_json::json!("bonsai-embeddings-tfidf"),
            "the response names the backend that answered: {first}"
        );
        let vectors = first["data"].as_array().expect("data");
        assert_eq!(vectors.len(), 2);
        let dim = vectors[0]["embedding"].as_array().expect("embedding").len();
        assert!(dim > 0 && dim <= TFIDF_MAX_FEATURES, "dim {dim}");
        let (_, second) = post_embeddings(&router, body).await;
        assert_eq!(
            first["data"], second["data"],
            "the fitted space never drifts"
        );

        // Other routes are untouched by the override.
        let health = router
            .clone()
            .oneshot(
                Request::get("/health")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(health.status(), StatusCode::OK);

        // ...and the bearer layer still guards the overridden route.
        let mut guarded = default_opts();
        guarded.bearer_token = Some("y".repeat(20));
        guarded.embeddings_override = Some(build_tfidf_embeddings_router(
            &corpus,
            Arc::new(InferenceMetrics::new()),
        ));
        let (status, _) = post_embeddings(&router_for(guarded), body).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    /// Built under the crate-wide env lock: `RouterOptions::default()` reads
    /// `OXI_ADMIN_TOKEN`, which sibling tests may be mutating (the lock is
    /// released before any `.await`).
    fn router_for(opts: HardeningOptions) -> Router {
        let _env = test_env::lock();
        let engine = InferenceEngine::new(Qwen3Config::tiny_test(), SamplingParams::default(), 42);
        let pool = EnginePool::new(vec![engine]);
        let pool_size = pool.size();
        let metrics = Arc::new(InferenceMetrics::new());
        let router_options = RouterOptions::default().with_auth(AdminAuthConfig::locked());
        let base = create_router_full(pool, None, metrics, router_options);
        harden_router(base, pool_size, &opts, "127.0.0.1")
    }

    #[tokio::test]
    async fn health_is_reachable_with_no_config() {
        let router = router_for(default_opts());
        let resp = router
            .oneshot(
                Request::get("/health")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn admin_is_refused_by_default() {
        let router = router_for(default_opts());
        let resp = router
            .oneshot(
                Request::get("/admin/status")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn bearer_auth_protects_inference_routes_when_configured() {
        let mut opts = default_opts();
        opts.bearer_token = Some("x".repeat(20));
        let router = router_for(opts);

        let unauth = router
            .clone()
            .oneshot(
                Request::get("/v1/models")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

        let health = router
            .oneshot(
                Request::get("/health")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(health.status(), StatusCode::OK, "/health must stay exempt");
    }

    /// cli-18: two consecutive unauthenticated requests must be
    /// rejected identically. **Not discriminating on its own**: both
    /// requests run sequentially through `oneshot`, so the sole permit
    /// (if bearer auth were ever mounted inside the admission stack)
    /// would already be released before the second request starts, and
    /// this assertion would hold under BOTH layer orderings. Kept as a
    /// cheap idempotency check; `unauthenticated_request_is_401_while_the_sole_permit_is_held`
    /// below is the test that actually proves the layer ordering.
    #[tokio::test]
    async fn unauthenticated_request_never_reaches_admission() {
        let mut opts = default_opts();
        opts.bearer_token = Some("x".repeat(20));
        opts.max_concurrent_requests = 1;
        let router = router_for(opts);

        for _ in 0..2 {
            let resp = router
                .clone()
                .oneshot(
                    Request::get("/v1/models")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        }
    }

    /// VERIFIER-ADDED (cli-18), ported from the discriminating version
    /// added to `crates/oxibonsai-serve/src/hardening.rs`'s
    /// `build_router_tests`: the sibling
    /// `unauthenticated_request_never_reaches_admission` above drives
    /// its two requests sequentially through `oneshot`, so the permit
    /// is released before the second request starts and the assertion
    /// holds under BOTH layer orderings -- it does not actually
    /// discriminate. Here a slow *authenticated* chat completion is
    /// kept in flight on a separate task, holding the sole concurrency
    /// permit (effective ceiling = `resolve_admission_limit(1, 1)` =
    /// `1`); an unauthenticated request issued while it runs must still
    /// be `401`. If bearer auth were mounted INSIDE the admission
    /// stack, `load_shed` would answer `503` instead.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unauthenticated_request_is_401_while_the_sole_permit_is_held() {
        let mut opts = default_opts();
        opts.bearer_token = Some("x".repeat(20));
        opts.max_concurrent_requests = 1;
        let router = router_for(opts);

        let slow_router = router.clone();
        let slow = tokio::spawn(async move {
            let payload = serde_json::json!({
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 500,
            });
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", "x".repeat(20)))
                .body(Body::from(payload.to_string()))
                .expect("request");
            slow_router.oneshot(req).await.map(|r| r.status())
        });

        tokio::time::sleep(Duration::from_millis(30)).await;
        let resp = router
            .oneshot(
                Request::get("/v1/models")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let slow_status = slow.await.expect("join").expect("slow response");
        assert_eq!(
            slow_status,
            StatusCode::OK,
            "the authenticated request must have been the one holding the permit"
        );
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "an unauthenticated request must be 401 even while the sole admission permit \
             is held (a 503 means bearer auth sits INSIDE the admission stack)"
        );
    }

    /// sec-20/perf-M1: the `503` an overloaded admission layer returns
    /// must carry `Retry-After` -- previously an omission (only the
    /// `429` rate-limit path below set it). Same permit-holding trick
    /// as `unauthenticated_request_is_401_while_the_sole_permit_is_held`,
    /// but with no bearer token configured, so the second request
    /// actually reaches (and is shed by) the admission layer instead of
    /// being rejected by auth first.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overloaded_request_gets_503_with_retry_after() {
        let mut opts = default_opts();
        opts.max_concurrent_requests = 1;
        let router = router_for(opts);

        let slow_router = router.clone();
        let slow = tokio::spawn(async move {
            let payload = serde_json::json!({
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 500,
            });
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .expect("request");
            slow_router.oneshot(req).await.map(|r| r.status())
        });

        tokio::time::sleep(Duration::from_millis(30)).await;
        let resp = router
            .oneshot(
                Request::get("/v1/models")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let slow_status = slow.await.expect("join").expect("slow response");
        assert_eq!(
            slow_status,
            StatusCode::OK,
            "the in-flight request must have been the one holding the sole permit"
        );
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            resp.headers().get("retry-after").is_some(),
            "a 503 from the admission layer must carry Retry-After"
        );
    }

    #[tokio::test]
    async fn rate_limit_returns_429_with_retry_after() {
        let mut opts = default_opts();
        opts.rate_limit_rpm = Some(60.0);
        opts.rate_limit_burst = 1.0;
        let router = router_for(opts);

        let first = router
            .clone()
            .oneshot(
                Request::get("/v1/models")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(first.status(), StatusCode::OK);

        let second = router
            .oneshot(
                Request::get("/v1/models")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(second.headers().get("retry-after").is_some());
    }

    #[tokio::test]
    async fn cors_preflight_bypasses_bearer_auth() {
        let mut opts = default_opts();
        opts.bearer_token = Some("x".repeat(20));
        opts.cors_origins = vec!["https://app.example.com".to_string()];
        let router = router_for(opts);

        let preflight = Request::builder()
            .method("OPTIONS")
            .uri("/v1/chat/completions")
            .header("origin", "https://app.example.com")
            .body(Body::empty())
            .expect("preflight request");
        let resp = router.oneshot(preflight).await.expect("response");
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "an OPTIONS preflight must bypass bearer auth entirely (SV-06)"
        );
    }

    #[tokio::test]
    async fn no_cors_configured_emits_no_cors_headers() {
        let router = router_for(default_opts());
        let resp = router
            .oneshot(
                Request::get("/health")
                    .header("origin", "https://anything.example.com")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert!(resp.headers().get("access-control-allow-origin").is_none());
    }

    #[tokio::test]
    async fn oversized_body_is_rejected_with_413() {
        let mut opts = default_opts();
        opts.max_body_bytes = 16;
        let router = router_for(opts);

        let body = Body::from(vec![b'a'; 4096]);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(body)
            .expect("request");
        let resp = router.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    /// FIX2-SERVE item 3, ported from the sibling test added to
    /// `crates/oxibonsai-serve/src/hardening.rs`'s `build_router_tests`:
    /// `DefaultBodyLimit` was reordered from inside the admission stack
    /// to outside it (still inside rate-limiting). **Not discriminating
    /// on its own** (same caveat as
    /// `unauthenticated_request_never_reaches_admission` above for a
    /// different layer pair): `axum::extract::DefaultBodyLimit` only
    /// inserts a request extension consulted later by the handler's own
    /// extractor and never rejects anything itself at either position,
    /// so this only proves the ceiling still applies once
    /// `max_concurrent_requests` is tightened to its minimum (1), and
    /// that the response is genuinely `413`, not a `408`/`504` timeout
    /// from admission's own `.timeout(..)`.
    #[tokio::test]
    async fn oversized_body_is_rejected_with_413_even_with_a_single_permit() {
        let mut opts = default_opts();
        opts.max_body_bytes = 16;
        opts.max_concurrent_requests = 1;
        let router = router_for(opts);

        let body = Body::from(vec![b'a'; 4096]);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(body)
            .expect("request");
        let resp = router.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    /// Wave-2.5 routing (3): the mirrored `Content-Length` precheck refuses
    /// a declared oversize with 413 WHILE the sole admission permit is held
    /// by a slow request — i.e. before admission. Without the guard the
    /// oversized request would queue behind the permit (load-shed 503).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn declared_oversize_is_413_while_the_sole_permit_is_held() {
        let mut opts = default_opts();
        opts.max_body_bytes = 256;
        opts.max_concurrent_requests = 1;
        let router = router_for(opts);

        let slow_router = router.clone();
        let slow = tokio::spawn(async move {
            let payload = serde_json::json!({
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 500,
            });
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .expect("request");
            slow_router.oneshot(req).await.map(|r| r.status())
        });
        tokio::time::sleep(Duration::from_millis(30)).await;

        let oversized = vec![b'a'; 4096];
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("content-length", oversized.len())
            .body(Body::from(oversized))
            .expect("request");
        let resp = router.oneshot(req).await.expect("response");
        let slow_status = slow.await.expect("join").expect("slow response");
        assert_eq!(slow_status, StatusCode::OK);
        assert_eq!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "a declared oversize must be refused before admission (not shed with 503)"
        );
    }
}

// ── RT-17: the pool's baseline follows the model's own sampling defaults ──

#[test]
fn serve_baseline_sampling_params_follow_the_gguf_defaults() {
    use oxibonsai_runtime::sampling::GgufSamplingDefaults;
    // A Bonsai 2 file declares 1.0 / 0.95 / 20 — the baseline takes them.
    let bonsai2 = GgufSamplingDefaults {
        temperature: Some(1.0),
        top_p: Some(0.95),
        top_k: Some(20),
        min_p: None,
    };
    let params = baseline_sampling_params(&bonsai2);
    assert!((params.temperature - 1.0).abs() < f32::EPSILON);
    assert!((params.top_p - 0.95).abs() < f32::EPSILON);
    assert_eq!(params.top_k, 20);
    assert!((params.repetition_penalty - 1.0).abs() < f32::EPSILON);
    // A partial declaration falls back field by field to the literals.
    let partial = GgufSamplingDefaults {
        temperature: Some(0.3),
        ..GgufSamplingDefaults::default()
    };
    let params = baseline_sampling_params(&partial);
    assert!((params.temperature - 0.3).abs() < f32::EPSILON);
    assert_eq!(params.top_k, 40);
    assert!((params.top_p - 0.9).abs() < f32::EPSILON);
}

// ── cli-11: server-wide chat defaults (pure text splicing) ──────────────────

fn all_defaults() -> ChatDefaults {
    ChatDefaults {
        enable_thinking: Some(false),
        reasoning_effort: Some("low".to_string()),
        tools_json: Some(
            r#"[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]"#
                .to_string(),
        ),
    }
}

#[test]
fn chat_defaults_are_spliced_in_front_and_keep_every_client_byte() {
    let body = br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}"#;
    let out = inject_chat_defaults(body, &all_defaults()).expect("defaults apply");
    let text = String::from_utf8(out.clone()).expect("utf8");
    // The client's own bytes follow the inserted fields unchanged.
    assert!(
        text.ends_with(r#""messages":[{"role":"user","content":"hi"}],"max_tokens":8}"#),
        "{text}"
    );
    // The tools text is inserted verbatim (key order preserved: "type" first).
    assert!(
        text.contains(r#""tools":[{"type": "function", "function": {"name": "get_weather""#),
        "{text}"
    );
    let parsed: serde_json::Value = serde_json::from_slice(&out).expect("still valid JSON");
    assert_eq!(parsed["enable_thinking"], serde_json::Value::Bool(false));
    assert_eq!(parsed["reasoning_effort"], "low");
    assert_eq!(parsed["max_tokens"], 8);
}

#[test]
fn a_request_carrying_its_own_contract_passes_through_untouched() {
    let body = br#"{"messages":[],"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"xhigh"},"tools":[{"function":{"name":"z"},"type":"function"}]}"#;
    assert_eq!(
        inject_chat_defaults(body, &all_defaults()),
        None,
        "every default is already covered by the request: forward the original bytes"
    );
    // Top-level spellings count as "carried" too.
    let body = br#"{"messages":[],"enable_thinking":true,"reasoning_effort":"medium","tools":[]}"#;
    assert_eq!(inject_chat_defaults(body, &all_defaults()), None);
}

#[test]
fn only_the_missing_defaults_are_added() {
    let body = br#"{"chat_template_kwargs":{"enable_thinking":true},"messages":[]}"#;
    let out = inject_chat_defaults(body, &all_defaults()).expect("effort + tools missing");
    let parsed: serde_json::Value = serde_json::from_slice(&out).expect("valid JSON");
    assert!(
        parsed.get("enable_thinking").is_none(),
        "the nested value stands alone"
    );
    assert_eq!(parsed["reasoning_effort"], "low");
    assert!(parsed["tools"].is_array());
}

#[test]
fn chat_defaults_handle_an_empty_object_and_ignore_non_objects() {
    let out = inject_chat_defaults(b"  {}", &all_defaults()).expect("empty object");
    let parsed: serde_json::Value =
        serde_json::from_slice(&out).expect("valid JSON (no trailing comma)");
    assert_eq!(parsed["reasoning_effort"], "low");
    assert_eq!(inject_chat_defaults(b"[1,2]", &all_defaults()), None);
    assert_eq!(inject_chat_defaults(b"not json", &all_defaults()), None);
    assert_eq!(inject_chat_defaults(b"{}", &ChatDefaults::default()), None);
}

mod chat_defaults_middleware_tests {
    use super::*;
    use axum::http::Request;
    use tower::ServiceExt;

    /// An echo route standing in for the chat handler: returns exactly the
    /// body bytes it received.
    fn echo_router(defaults: ChatDefaults) -> Router {
        Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(|body: axum::body::Bytes| async move { body }),
            )
            .route(
                "/v1/embeddings",
                axum::routing::post(|body: axum::body::Bytes| async move { body }),
            )
            .layer(axum::middleware::from_fn_with_state(
                Arc::new(ChatDefaultsState {
                    defaults,
                    max_body_bytes: 1 << 20,
                }),
                chat_defaults_mw,
            ))
    }

    async fn post(router: Router, path: &str, body: &'static [u8]) -> Vec<u8> {
        let req = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .header("content-length", body.len())
            .body(Body::from(body))
            .expect("request");
        let resp = router.oneshot(req).await.expect("response");
        axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("body")
            .to_vec()
    }

    #[tokio::test]
    async fn a_request_with_its_own_kwargs_and_tools_reaches_the_handler_byte_identical() {
        let body: &'static [u8] = br#"{ "messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "medium"}, "tools": [{"type": "function", "function": {"name": "f", "parameters": {"b": 1, "a": 2}}}] }"#;
        let echoed = post(echo_router(all_defaults()), "/v1/chat/completions", body).await;
        assert_eq!(echoed, body.to_vec());
    }

    #[tokio::test]
    async fn a_bare_request_inherits_the_server_defaults() {
        let body: &'static [u8] = br#"{"messages":[{"role":"user","content":"x"}]}"#;
        let echoed = post(echo_router(all_defaults()), "/v1/chat/completions", body).await;
        let parsed: serde_json::Value = serde_json::from_slice(&echoed).expect("valid JSON");
        assert_eq!(parsed["enable_thinking"], serde_json::Value::Bool(false));
        assert_eq!(parsed["reasoning_effort"], "low");
        assert!(parsed["tools"].is_array());
    }

    #[tokio::test]
    async fn other_routes_are_never_rewritten() {
        let body: &'static [u8] = br#"{"input":"x"}"#;
        let echoed = post(echo_router(all_defaults()), "/v1/embeddings", body).await;
        assert_eq!(echoed, body.to_vec());
    }
}

// ── B2-13 LEAD ITEM: the serving tokenizer ladder, three file kinds ─────────

mod serving_tokenizer_tests {
    use super::*;
    use crate::cli::test_fixtures::{
        byte_level_tokenizer_json, tokenizer_host_gguf, BYTE_LEVEL_VOCAB, THINKING_TEMPLATE,
        THINK_CLOSE_ID, THINK_OPEN_ID, TOOL_CALL_CLOSE_ID, TOOL_CALL_OPEN_ID,
    };
    use oxibonsai_runtime::config::ResolvedChatTemplate;

    struct Fixture {
        dir: std::path::PathBuf,
        model: String,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A model file (and optionally a `tokenizer.json` beside it) in a
    /// fresh directory.
    fn fixture(tag: &str, gguf: &[u8], tokenizer_json: bool) -> Fixture {
        let dir = crate::cli::test_fixtures::scratch_dir(&format!("serve_tok_{tag}"));
        let model = dir.join("Model.gguf");
        std::fs::write(&model, gguf).expect("write model");
        if tokenizer_json {
            std::fs::write(dir.join("tokenizer.json"), byte_level_tokenizer_json())
                .expect("write tokenizer.json");
        }
        Fixture {
            model: model.to_string_lossy().into_owned(),
            dir,
        }
    }

    fn load(fix: &Fixture) -> oxibonsai_runtime::TokenizerBridge {
        let mmap =
            oxibonsai_core::gguf::reader::mmap_gguf_file(Path::new(&fix.model)).expect("map");
        let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap).expect("parse");
        let (tok, _) = load_serving_tokenizer(None, &fix.model, &gguf).expect("resolve");
        tok.expect("a tokenizer must be found")
    }

    fn assert_model_ids(tok: &oxibonsai_runtime::TokenizerBridge) {
        assert_eq!(tok.think_open_id(), Some(THINK_OPEN_ID));
        assert_eq!(tok.think_close_id(), Some(THINK_CLOSE_ID));
        assert_eq!(tok.tool_call_open_id(), Some(TOOL_CALL_OPEN_ID));
        assert_eq!(tok.tool_call_close_id(), Some(TOOL_CALL_CLOSE_ID));
    }

    #[test]
    fn embedded_vocabulary_and_template_come_from_the_gguf() {
        let fix = fixture(
            "embedded",
            &tokenizer_host_gguf(BYTE_LEVEL_VOCAB, Some(THINKING_TEMPLATE), true),
            false,
        );
        let tok = load(&fix);
        assert_eq!(tok.vocab_size() as u64, BYTE_LEVEL_VOCAB);
        assert!(matches!(
            tok.resolved_chat_template(),
            ResolvedChatTemplate::Jinja(_)
        ));
        assert_model_ids(&tok);
    }

    #[test]
    fn a_tokenizer_json_gets_the_gguf_template_attached() {
        let fix = fixture(
            "json_plus_template",
            &tokenizer_host_gguf(BYTE_LEVEL_VOCAB, Some(THINKING_TEMPLATE), false),
            true,
        );
        let tok = load(&fix);
        let template = tok.resolved_chat_template();
        assert!(
            matches!(template, ResolvedChatTemplate::Jinja(_)),
            "the GGUF's own template must win over the ChatML fallback"
        );
        // It is THIS template: it opens a think block by default.
        assert!(bonsai2::default_enable_thinking(&template).expect("render"));
        assert_model_ids(&tok);
    }

    #[test]
    fn a_tokenizer_json_with_no_gguf_template_falls_back_to_chatml() {
        let fix = fixture(
            "json_no_template",
            &tokenizer_host_gguf(BYTE_LEVEL_VOCAB, None, false),
            true,
        );
        let tok = load(&fix);
        assert!(matches!(
            tok.resolved_chat_template(),
            ResolvedChatTemplate::Canned(_)
        ));
        assert_model_ids(&tok);
    }

    #[test]
    fn a_vocabulary_mismatch_is_refused_for_an_explicit_tokenizer() {
        // TOK-08 on serve: the model declares 300 rows, the explicit
        // tokenizer has 262 — a different BPE, a hard error.
        let fix = fixture("mismatch", &tokenizer_host_gguf(300, None, false), true);
        let explicit = fix
            .dir
            .join("tokenizer.json")
            .to_string_lossy()
            .into_owned();
        let mmap =
            oxibonsai_core::gguf::reader::mmap_gguf_file(Path::new(&fix.model)).expect("map");
        let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap).expect("parse");
        let err = load_serving_tokenizer(Some(&explicit), &fix.model, &gguf)
            .err()
            .expect("mismatch must be refused");
        assert!(err.to_string().contains("vocabulary mismatch"), "{err}");
    }

    /// The real Bonsai 2 27B header (`OXI_BONSAI2_PQ2_GGUF`): the serving
    /// tokenizer is the GGUF-embedded one, its template is the GGUF's own,
    /// and the chat-contract ids are the vocabulary's (design Appendix A.1).
    #[test]
    fn the_real_27b_file_serves_its_own_template_and_ids() {
        let Some(path) = crate::cli::test_fixtures::env_path(
            "OXI_BONSAI2_PQ2_GGUF",
            "the real Ternary-Bonsai-2-27B-PQ2_0.gguf",
        ) else {
            return;
        };
        let _real = crate::cli::test_fixtures::real_model_lock();
        let model = path.to_string_lossy().into_owned();
        let mmap = oxibonsai_core::gguf::reader::mmap_gguf_file(&path).expect("map");
        let gguf = oxibonsai_core::gguf::reader::GgufFile::parse(&mmap).expect("parse");
        let (tok, _) = load_serving_tokenizer(None, &model, &gguf).expect("resolve");
        let tok = tok.expect("the 27B embeds its tokenizer");
        assert_eq!(tok.vocab_size(), 248_320);
        assert!(matches!(
            tok.resolved_chat_template(),
            ResolvedChatTemplate::Jinja(_)
        ));
        assert_eq!(tok.think_open_id(), Some(248_068));
        assert_eq!(tok.think_close_id(), Some(248_069));
        assert_eq!(tok.tool_call_open_id(), Some(248_058));
        assert_eq!(tok.tool_call_close_id(), Some(248_059));
        assert!(bonsai2::default_enable_thinking(&tok.resolved_chat_template()).expect("render"));
    }
}

// ── ENGINE-SEAM (4): a hybrid model's /v1/embeddings is an honest 501 ───────

/// Real 27B (`OXI_BONSAI2_PQ2_GGUF`): `build_embedder` refuses the hybrid
/// with the typed `NOT_A_DENSE_MODEL` error, logs the known limitation at
/// info, and the served router answers `/v1/embeddings` with 501.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hybrid_model_serves_embeddings_as_an_honest_501_with_an_info_log() {
    use axum::http::Request;
    use tower::ServiceExt;
    let Some(path) = crate::cli::test_fixtures::env_path(
        "OXI_BONSAI2_PQ2_GGUF",
        "the real Ternary-Bonsai-2-27B-PQ2_0.gguf",
    ) else {
        return;
    };
    let (router, events) = {
        let _real = crate::cli::test_fixtures::real_model_lock();
        let model = path.to_string_lossy().into_owned();
        let bytes = ModelSource::open(&model, false).expect("map").into_static();
        let gguf: &'static oxibonsai_core::gguf::reader::GgufFile<'static> = Box::leak(Box::new(
            oxibonsai_core::gguf::reader::GgufFile::parse(bytes).expect("parse"),
        ));
        let params = default_sampling_params();
        let built = oxibonsai_runtime::engine_pool::build_pool_from_static_gguf_with_rope(
            gguf,
            params.clone(),
            42,
            64,
            Some(1),
            oxibonsai_runtime::engine_seam::Backend::Cpu,
            oxibonsai_core::config::RopeScalingOverride::Auto,
        )
        .expect("the hybrid pool loads on the CPU tier");
        assert!(built.hybrid);
        let (tok, _) = load_serving_tokenizer(None, &model, gguf).expect("tokenizer");
        let capture = crate::cli::test_fixtures::CapturedEvents::default();
        let embedder = tracing::subscriber::with_default(capture.clone(), || {
            build_embedder(
                &built,
                tok,
                EmbeddingEngineLoad {
                    params,
                    seed: 42,
                    max_seq_len: 64,
                    backend: oxibonsai_runtime::engine_seam::Backend::Cpu,
                    rope_scaling: oxibonsai_runtime::config::RopeScalingMode::Auto,
                },
            )
        });
        assert!(embedder.is_none(), "a hybrid model has no embedder yet");
        let router = {
            let _env = test_env::lock();
            create_router_full(
                Arc::clone(&built.pool),
                None,
                Arc::new(oxibonsai_runtime::InferenceMetrics::new()),
                RouterOptions::default()
                    .with_auth(AdminAuthConfig::locked())
                    .with_embedder(embedder),
            )
        };
        (router, capture.events())
    };
    assert!(
        events.iter().any(|e| e.contains("[INFO]")
            && e.contains("embeddings are not supported for this model yet")
            && e.contains("NOT_A_DENSE_MODEL")),
        "the known limitation must be logged at info: {events:?}"
    );
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"input":"hello","model":"m"}"#))
        .expect("request");
    let resp = router.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
}
