//! Unit tests for `util.rs`, split into a sibling file to keep
//! `util.rs` under the 2000-line policy limit (declared there via
//! `#[path]`, so `super` still names that module).

use super::{
    clamp_generation_budget, missing_tokenizer_warning, parse_flat_toml_sections,
    read_prompt_stdin, reject_penalties_with_constrained_decode, resolve_backend, resolve_f32,
    resolve_rope_scaling, resolve_str, resolve_tokenizer, resolve_usize, strip_quant_suffix,
    tokenizer_candidates, toml_f32, toml_str, toml_usize, RawTomlSections,
};
use oxibonsai_runtime::config::RopeScalingMode;
use oxibonsai_runtime::engine_seam::Backend;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Helper: write an empty `tokenizer.json` at the given path, creating
/// any missing parent directories.
fn touch_tokenizer(path: &std::path::Path) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create_dir_all");
    }
    fs::write(path, b"{}").expect("write tokenizer.json");
}

#[test]
fn resolve_tokenizer_finds_in_same_dir() {
    let tmp = TempDir::new().expect("tempdir");
    let model_dir = tmp.path().join("models");
    fs::create_dir_all(&model_dir).expect("create model_dir");
    let model_path = model_dir.join("Foo-Q2_0.gguf");
    fs::write(&model_path, b"").expect("touch model");
    touch_tokenizer(&model_dir.join("tokenizer.json"));

    let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
    let found = lookup.found.as_deref().expect("expected to find tokenizer");
    assert_eq!(
        PathBuf::from(found),
        model_dir.join("tokenizer.json"),
        "should locate tokenizer in the same dir as the model"
    );
}

#[test]
fn resolve_tokenizer_finds_in_parent_dir() {
    let tmp = TempDir::new().expect("tempdir");
    let model_dir = tmp.path().join("models").join("variant");
    fs::create_dir_all(&model_dir).expect("create model_dir");
    let model_path = model_dir.join("Foo-Q2_0.gguf");
    fs::write(&model_path, b"").expect("touch model");
    // Place tokenizer in the parent directory only.
    let parent_tokenizer = tmp.path().join("models").join("tokenizer.json");
    touch_tokenizer(&parent_tokenizer);

    let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
    let found = lookup.found.as_deref().expect("expected to find tokenizer");
    // Either the literal `..` candidate or the canonicalized `models/tokenizer.json`
    // candidate is acceptable; both refer to the same file.
    let found_path = PathBuf::from(found);
    let canon_found = fs::canonicalize(&found_path).expect("canonicalize found");
    let canon_target = fs::canonicalize(&parent_tokenizer).expect("canonicalize target");
    assert_eq!(
        canon_found, canon_target,
        "should locate tokenizer in the model's parent directory"
    );
}

#[test]
fn resolve_tokenizer_finds_via_unpacked_sibling() {
    let tmp = TempDir::new().expect("tempdir");
    let model_dir = tmp.path().join("models");
    fs::create_dir_all(&model_dir).expect("create model_dir");
    let model_path = model_dir.join("Ternary-Bonsai-8B-Q2_0.gguf");
    fs::write(&model_path, b"").expect("touch model");
    // Tokenizer only lives in the sibling unpacked directory.
    let unpacked = model_dir.join("Ternary-Bonsai-8B-unpacked");
    touch_tokenizer(&unpacked.join("tokenizer.json"));

    let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
    let found = lookup.found.as_deref().expect("expected to find tokenizer");
    assert_eq!(
        PathBuf::from(found),
        unpacked.join("tokenizer.json"),
        "should locate tokenizer via <base>-unpacked sibling directory"
    );
}

#[test]
fn resolve_tokenizer_strips_quant_suffix_for_sibling_lookup() {
    // Verifies the candidate list (without filesystem) for the
    // `Foo-Q2_0.gguf` case includes Foo/, Foo-unpacked/, Foo-ONNX/.
    let model_path = PathBuf::from("models/Foo-Q2_0.gguf");
    let candidates = tokenizer_candidates(&model_path);
    let candidate_strs: Vec<String> = candidates
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    let expected = [
        "models/Foo/tokenizer.json",
        "models/Foo-unpacked/tokenizer.json",
        "models/Foo-ONNX/tokenizer.json",
    ];
    for needle in expected {
        assert!(
            candidate_strs.iter().any(|c| c == needle),
            "missing expected candidate {needle}; got {candidate_strs:?}"
        );
    }
}

#[test]
fn resolve_tokenizer_records_searched_paths_when_missing() {
    let tmp = TempDir::new().expect("tempdir");
    let model_dir = tmp.path().join("models");
    fs::create_dir_all(&model_dir).expect("create model_dir");
    let model_path = model_dir.join("Ternary-Bonsai-8B-Q2_0.gguf");
    fs::write(&model_path, b"").expect("touch model");

    let lookup = resolve_tokenizer(None, model_path.to_str().expect("utf8"));
    assert!(
        lookup.found.is_none(),
        "should not find tokenizer in empty tree"
    );
    assert!(
        !lookup.searched.is_empty(),
        "searched list must be populated when nothing is found"
    );
    // Confirm at least the "same dir" candidate is recorded.
    assert!(
        lookup
            .searched
            .iter()
            .any(|p| p == &model_dir.join("tokenizer.json")),
        "searched list should include the same-directory candidate"
    );
    // Warning text must mention every searched path and both remedies.
    let warning = missing_tokenizer_warning(&lookup.searched);
    for path in &lookup.searched {
        assert!(
            warning.contains(&path.display().to_string()),
            "warning should list {}, got: {warning}",
            path.display()
        );
    }
    assert!(
        warning.contains("--tokenizer"),
        "warning must mention --tokenizer remedy"
    );
    assert!(
        warning.contains("download_tokenizer.sh"),
        "warning must mention download_tokenizer.sh remedy"
    );
}

#[test]
fn resolve_tokenizer_explicit_override_skips_search() {
    let lookup = resolve_tokenizer(Some("/custom/path/tokenizer.json"), "models/foo.gguf");
    assert_eq!(
        lookup.found.as_deref(),
        Some("/custom/path/tokenizer.json"),
        "explicit override must be returned verbatim"
    );
    assert!(
        lookup.searched.is_empty(),
        "explicit override must not trigger a filesystem search"
    );
}

#[test]
fn strip_quant_suffix_handles_known_formats() {
    assert_eq!(
        strip_quant_suffix("Ternary-Bonsai-8B-Q2_0"),
        "Ternary-Bonsai-8B"
    );
    assert_eq!(strip_quant_suffix("Foo-Q1_0"), "Foo");
    assert_eq!(strip_quant_suffix("Foo-Q4_K_M"), "Foo");
    assert_eq!(strip_quant_suffix("Foo-Q8_0"), "Foo");
    assert_eq!(strip_quant_suffix("Foo-F16"), "Foo");
    assert_eq!(strip_quant_suffix("Foo-BF16"), "Foo");
    assert_eq!(strip_quant_suffix("Foo-F32"), "Foo");
    // Non-quant suffix should be left alone.
    assert_eq!(strip_quant_suffix("Foo-bar"), "Foo-bar");
    assert_eq!(strip_quant_suffix("Foo"), "Foo");
}

#[test]
fn tokenizer_candidates_includes_top_level_models_dir() {
    let model_path = PathBuf::from("models/sub/dir/Foo-Q2_0.gguf");
    let candidates = tokenizer_candidates(&model_path);
    let candidate_strs: Vec<String> = candidates
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    assert!(
        candidate_strs.iter().any(|c| c == "models/tokenizer.json"),
        "expected top-level models/tokenizer.json candidate, got {candidate_strs:?}"
    );
}

// ── read_prompt_stdin (cli-M4) ──────────────────────────────────────

#[test]
fn read_prompt_stdin_rejects_empty_input_type_check() {
    // A full stdin-redirection test belongs in an integration test
    // (tests/cli_surface_tests.rs); this just locks in the new
    // `Result` signature so callers must handle the error instead of
    // getting a silently-possibly-truncated `String` back.
    fn assert_is_result(_: fn() -> anyhow::Result<String>) {}
    assert_is_result(read_prompt_stdin);
}

// ── config raw-TOML scanner (cli-04) ────────────────────────────────

#[test]
fn parse_flat_toml_sections_accepts_well_formed_config() {
    let toml = r#"
        [server]
        host = "0.0.0.0"
        port = 9090

        [sampling]
        temperature = 0.5
        top_k = 20
        top_p = 0.95
        repetition_penalty = 1.05
        max_tokens = 256

        [model]
        model_path = "models/foo.gguf"
        tokenizer_path = "models/tokenizer.json"
        max_seq_len = 8192

        [observability]
        log_level = "debug"
        json_logs = true
    "#;
    let sections = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"))
        .expect("well-formed config must be accepted");
    assert_eq!(
        toml_str(&sections, "server", "host").as_deref(),
        Some("0.0.0.0")
    );
    assert_eq!(toml_usize(&sections, "server", "port"), Some(9090));
    assert_eq!(toml_f32(&sections, "sampling", "temperature"), Some(0.5));
    assert_eq!(
        toml_f32(&sections, "sampling", "repetition_penalty"),
        Some(1.05)
    );
    assert_eq!(
        toml_str(&sections, "model", "model_path").as_deref(),
        Some("models/foo.gguf")
    );
    assert_eq!(
        toml_str(&sections, "observability", "log_level").as_deref(),
        Some("debug")
    );
}

#[test]
fn parse_flat_toml_sections_rejects_typo_d_key() {
    let toml = "[sampling]\ntemperture = 0.5\n";
    let result = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("temperture"),
        "error should name the bad key: {msg}"
    );
}

#[test]
fn parse_flat_toml_sections_rejects_unknown_section() {
    let toml = "[imagen]\ndit_path = \"x\"\n";
    let result = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("imagen"),
        "error should name the bad section: {msg}"
    );
}

#[test]
fn parse_flat_toml_sections_ignores_comments_and_blank_lines() {
    let toml = "# a comment\n\n[server]\n# host is bound here\nhost = \"127.0.0.1\" # trailing\n";
    let sections = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"))
        .expect("comments must not break parsing");
    assert_eq!(
        toml_str(&sections, "server", "host").as_deref(),
        Some("127.0.0.1")
    );
}

#[test]
fn parse_flat_toml_sections_does_not_false_positive_on_hash_in_string() {
    let toml = "[observability]\nlog_level = \"info#not-a-comment\"\n";
    let sections = parse_flat_toml_sections(toml, std::path::Path::new("test.toml"))
        .expect("a '#' inside a quoted string must not be treated as a comment");
    assert_eq!(
        toml_str(&sections, "observability", "log_level").as_deref(),
        Some("info#not-a-comment")
    );
}

#[test]
fn parse_flat_toml_sections_absent_key_returns_none_not_a_default() {
    // The whole point of this scanner (vs. the typed, `#[serde(default)]`
    // struct): a key that was never in the file must read back as
    // `None`, not silently produce that struct's own built-in default.
    let toml = "[sampling]\ntemperature = 0.5\n";
    let sections =
        parse_flat_toml_sections(toml, std::path::Path::new("test.toml")).expect("valid config");
    assert_eq!(toml_f32(&sections, "sampling", "repetition_penalty"), None);
}

// ── resolve_* precedence (cli-04 / orchestrator P0 addendum) ────────

#[test]
fn resolve_f32_prefers_explicit_cli_value_over_config() {
    let mut sections = RawTomlSections::new();
    sections
        .entry("sampling".to_string())
        .or_default()
        .insert("repetition_penalty".to_string(), "1.4".to_string());
    let resolved = resolve_f32(Some(2.0), &sections, "sampling", "repetition_penalty", 1.0);
    assert_eq!(resolved, 2.0, "an explicit CLI flag must win over --config");
}

#[test]
fn resolve_f32_falls_back_to_config_value() {
    let mut sections = RawTomlSections::new();
    sections
        .entry("sampling".to_string())
        .or_default()
        .insert("temperature".to_string(), "0.3".to_string());
    let resolved = resolve_f32(None, &sections, "sampling", "temperature", 0.7);
    assert_eq!(resolved, 0.3);
}

#[test]
fn resolve_f32_repetition_penalty_never_silently_becomes_1_1() {
    // The whole point of routing repetition_penalty through the raw
    // scanner instead of the typed `SamplingConfig` (whose built-in default
    // was 1.1 before RT-24 made it 1.0): an empty (or unrelated) config
    // must resolve to the CLI's own safe default (1.0), never whatever
    // that struct's built-in value happens to be.
    let sections = RawTomlSections::new();
    let resolved = resolve_f32(None, &sections, "sampling", "repetition_penalty", 1.0);
    assert_eq!(resolved, 1.0);
}

#[test]
fn resolve_usize_and_str_use_hardcoded_default_with_no_config() {
    let sections = RawTomlSections::new();
    assert_eq!(
        resolve_usize(None, &sections, "sampling", "max_tokens", 256),
        256
    );
    assert_eq!(resolve_str(None, &sections, "model", "model_path"), None);
}

// ── reject_penalties_with_constrained_decode (the --stop/--grammar +
// penalty silent-drop finding) ───────────────────────────────────────

#[test]
fn constrained_decode_with_default_penalties_is_allowed() {
    reject_penalties_with_constrained_decode(true, 1.0, 0.0, 0.0)
        .expect("all-default penalties never conflict with --grammar/--stop");
}

#[test]
fn non_constrained_decode_allows_any_penalty() {
    // The fast path (`generate`/`generate_streaming_sync`) DOES apply
    // penalties via `sample_with_history`, so outside a
    // grammar/stop-checking loop every value is fine.
    reject_penalties_with_constrained_decode(false, 1.4, 0.5, -0.5)
        .expect("the fast path applies penalties; no combination is rejected");
}

#[test]
fn constrained_decode_rejects_non_default_repetition_penalty() {
    let err = reject_penalties_with_constrained_decode(true, 1.2, 0.0, 0.0)
        .expect_err("a non-default repetition penalty must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("--repetition-penalty"), "got: {msg}");
}

#[test]
fn constrained_decode_rejects_non_default_frequency_penalty() {
    let err = reject_penalties_with_constrained_decode(true, 1.0, 0.3, 0.0)
        .expect_err("a non-default frequency penalty must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("--frequency-penalty"), "got: {msg}");
}

#[test]
fn constrained_decode_rejects_non_default_presence_penalty() {
    let err = reject_penalties_with_constrained_decode(true, 1.0, 0.0, 0.3)
        .expect_err("a non-default presence penalty must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("--presence-penalty"), "got: {msg}");
}

#[test]
fn constrained_decode_rejection_names_every_offending_flag() {
    let err = reject_penalties_with_constrained_decode(true, 1.5, 0.2, 0.1)
        .expect_err("all three penalties are non-default");
    let msg = err.to_string();
    assert!(msg.contains("--repetition-penalty"), "got: {msg}");
    assert!(msg.contains("--frequency-penalty"), "got: {msg}");
    assert!(msg.contains("--presence-penalty"), "got: {msg}");
}

// ── resolve_backend / resolve_rope_scaling ──────────────────────────────

#[test]
fn resolve_backend_prefers_explicit_cli_value() {
    let sections = RawTomlSections::new();
    let backend =
        resolve_backend(Some(Backend::Cpu), &sections, "model", "backend").expect("resolves");
    assert_eq!(backend, Backend::Cpu);
}

#[test]
fn resolve_backend_falls_back_to_toml_then_default() {
    let toml = "[model]\nbackend = \"metal\"\n";
    let sections =
        parse_flat_toml_sections(toml, std::path::Path::new("test.toml")).expect("well-formed");
    assert_eq!(
        resolve_backend(None, &sections, "model", "backend").expect("resolves"),
        Backend::Metal
    );

    let empty = RawTomlSections::new();
    assert_eq!(
        resolve_backend(None, &empty, "model", "backend").expect("resolves"),
        Backend::Auto,
        "no flag and no config key must fall back to the engine's own default"
    );
}

#[test]
fn resolve_backend_rejects_an_invalid_toml_value() {
    let toml = "[model]\nbackend = \"quantum\"\n";
    let sections =
        parse_flat_toml_sections(toml, std::path::Path::new("test.toml")).expect("well-formed");
    let err = resolve_backend(None, &sections, "model", "backend")
        .expect_err("must reject an unknown backend name");
    assert!(err.to_string().contains("quantum"));
}

#[test]
fn resolve_rope_scaling_prefers_explicit_cli_value() {
    let sections = RawTomlSections::new();
    let mode = resolve_rope_scaling(
        Some(RopeScalingMode::Off),
        &sections,
        "model",
        "rope_scaling",
    )
    .expect("resolves");
    assert_eq!(mode, RopeScalingMode::Off);
}

#[test]
fn resolve_rope_scaling_falls_back_to_toml_then_default() {
    let toml = "[model]\nrope_scaling = \"on\"\n";
    let sections =
        parse_flat_toml_sections(toml, std::path::Path::new("test.toml")).expect("well-formed");
    assert_eq!(
        resolve_rope_scaling(None, &sections, "model", "rope_scaling").expect("resolves"),
        RopeScalingMode::On
    );

    let empty = RawTomlSections::new();
    assert_eq!(
        resolve_rope_scaling(None, &empty, "model", "rope_scaling").expect("resolves"),
        RopeScalingMode::Auto
    );
}

#[test]
fn resolve_rope_scaling_rejects_an_invalid_toml_value() {
    let toml = "[model]\nrope_scaling = \"yarn\"\n";
    let sections =
        parse_flat_toml_sections(toml, std::path::Path::new("test.toml")).expect("well-formed");
    let err = resolve_rope_scaling(None, &sections, "model", "rope_scaling")
        .expect_err("must reject an unknown rope-scaling mode");
    assert!(err.to_string().contains("yarn"));
}

// ── config surface: [imagen], the new keys, validated_opt ──────────────────

#[test]
fn parse_flat_toml_sections_rejects_a_section_outside_the_schema() {
    let err =
        parse_flat_toml_sections("[vision]\nmmproj = \"x\"\n", std::path::Path::new("t.toml"))
            .expect_err("[vision] is not part of the schema");
    let msg = err.to_string();
    assert!(msg.contains("unknown section [vision]"), "{msg}");
    assert!(
        msg.contains("imagen"),
        "the known-sections list includes [imagen]: {msg}"
    );
}

#[test]
fn imagen_is_a_known_section_with_exactly_its_documented_keys() {
    let toml = "[imagen]\nmodel_path = \"dit.gguf\"\nwidth = 768\nheight = 512\nsteps = 8\n\
                guidance_scale = 3.5\nseed = 7\noutput_dir = \"out\"\n";
    let sections = parse_flat_toml_sections(toml, std::path::Path::new("t.toml"))
        .expect("every documented [imagen] key parses");
    assert_eq!(
        toml_str(&sections, "imagen", "model_path").as_deref(),
        Some("dit.gguf")
    );
    assert_eq!(toml_usize(&sections, "imagen", "width"), Some(768));
    assert_eq!(super::toml_u64(&sections, "imagen", "seed"), Some(7));
    let err = parse_flat_toml_sections(
        "[imagen]\ndit_path = \"x\"\n",
        std::path::Path::new("t.toml"),
    )
    .expect_err("an undocumented key");
    assert!(
        err.to_string()
            .contains("unknown key 'dit_path' in section [imagen]"),
        "{err}"
    );
}

#[test]
fn the_wave_4b_model_and_server_keys_parse() {
    let toml = "[model]\nenable_thinking = false\nprefill_chunk = 256\nptq1_transcode = true\n\
                reasoning_effort = \"low\"\n[server]\nenable_ui = true\nmax_output_tokens = 2048\n\
                cuda_device = 1\n";
    let sections = parse_flat_toml_sections(toml, std::path::Path::new("t.toml")).expect("parses");
    assert_eq!(
        super::toml_bool(&sections, "model", "enable_thinking"),
        Some(false)
    );
    assert_eq!(toml_usize(&sections, "model", "prefill_chunk"), Some(256));
    assert_eq!(
        super::toml_bool(&sections, "model", "ptq1_transcode"),
        Some(true)
    );
    assert_eq!(
        super::toml_bool(&sections, "server", "enable_ui"),
        Some(true)
    );
    assert_eq!(
        toml_usize(&sections, "server", "max_output_tokens"),
        Some(2048)
    );
    assert_eq!(super::toml_u32(&sections, "server", "cuda_device"), Some(1));
    assert_eq!(
        super::toml_bool(&sections, "model", "reasoning_effort"),
        None,
        "not a bool"
    );
}

#[test]
fn validated_opt_passes_none_through_and_names_a_bad_value() {
    let ok: Option<f32> = super::validated_opt(None, super::super::args::validate_temperature)
        .expect("None is never validated");
    assert_eq!(ok, None);
    let good = super::validated_opt(Some(0.5f32), super::super::args::validate_temperature)
        .expect("in range");
    assert_eq!(good, Some(0.5));
    let err = super::validated_opt(Some(-5.0f32), super::super::args::validate_temperature)
        .expect_err("negative");
    assert!(err.to_string().contains("temperature"), "{err}");
}

#[test]
fn env_var_guard_restores_the_prior_value_even_when_it_was_unset() {
    use super::test_env::{lock, EnvVarGuard};
    const KEY: &str = "OXIBONSAI_UTIL_TEST_ENV_GUARD_PROBE";
    let _env = lock();
    {
        let _clear = EnvVarGuard::remove(KEY);
        {
            let _set = EnvVarGuard::set(KEY, "inner");
            assert_eq!(std::env::var(KEY).ok().as_deref(), Some("inner"));
        }
        assert_eq!(std::env::var(KEY).ok(), None, "restored to unset");
        let _outer = EnvVarGuard::set(KEY, "outer");
        {
            let _inner = EnvVarGuard::set(KEY, "inner");
        }
        assert_eq!(
            std::env::var(KEY).ok().as_deref(),
            Some("outer"),
            "restored to the prior value"
        );
    }
    assert_eq!(std::env::var(KEY).ok(), None);
}

// ── clamp_generation_budget ──────────────────────────────────────────────

#[test]
fn clamp_generation_budget_is_a_no_op_when_everything_fits() {
    assert_eq!(clamp_generation_budget(5, 10, 100).expect("fits"), 10);
}

#[test]
fn clamp_generation_budget_shrinks_to_what_remains() {
    // prompt 10, ctx 16: only 6 slots remain, even though 64 were requested.
    assert_eq!(clamp_generation_budget(10, 64, 16).expect("clamped"), 6);
}

#[test]
fn clamp_generation_budget_allows_exactly_filling_the_context() {
    assert_eq!(clamp_generation_budget(10, 6, 16).expect("exact fit"), 6);
}

#[test]
fn clamp_generation_budget_can_clamp_to_zero_when_the_prompt_fills_the_context() {
    assert_eq!(clamp_generation_budget(16, 8, 16).expect("zero left"), 0);
}

#[test]
fn clamp_generation_budget_refuses_a_prompt_that_alone_overflows() {
    let err = clamp_generation_budget(17, 1, 16).expect_err("prompt alone too long");
    let msg = err.to_string();
    assert!(msg.contains("17"), "{msg}");
    assert!(msg.contains("16"), "{msg}");
}
