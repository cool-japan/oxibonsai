//! Integration tests for oxibonsai-serve argument parsing.

use oxibonsai_serve::args::{parse_args_from, ParseError, ServerArgs};

// ─── Helpers ───────────────────────────────────────────────────────────────

fn argv(flags: &[&str]) -> Vec<String> {
    // Mimic a real argv: first element is the program name.
    std::iter::once("oxibonsai-serve")
        .chain(flags.iter().copied())
        .map(str::to_string)
        .collect()
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn default_args_sensible() {
    let result = parse_args_from(&argv(&[]))
        .expect("empty argv should parse without error")
        .expect("empty argv should not trigger help/version");

    assert_eq!(result.port, 8080);
    assert_eq!(result.max_tokens, 256);
    assert!((result.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(result.seed, 42);
    assert_eq!(result.host, "127.0.0.1");
    assert_eq!(result.log_level, "info");
    assert!(result.model_path.is_none());
    assert!(result.tokenizer_path.is_none());
}

#[test]
fn parse_host_port() {
    let result = parse_args_from(&argv(&["--host", "127.0.0.1", "--port", "3000"]))
        .expect("should parse without error")
        .expect("should not trigger help/version");

    assert_eq!(result.host, "127.0.0.1");
    assert_eq!(result.port, 3000);
}

#[test]
fn parse_model_path() {
    let result = parse_args_from(&argv(&["--model", "/path/to/model.gguf"]))
        .expect("should parse without error")
        .expect("should not trigger help/version");

    assert_eq!(result.model_path, Some("/path/to/model.gguf".to_string()));
}

#[test]
fn parse_temperature() {
    let result = parse_args_from(&argv(&["--temperature", "0.5"]))
        .expect("should parse without error")
        .expect("should not trigger help/version");

    assert!((result.temperature - 0.5_f32).abs() < f32::EPSILON);
}

#[test]
fn parse_seed() {
    let result = parse_args_from(&argv(&["--seed", "1234"]))
        .expect("should parse without error")
        .expect("should not trigger help/version");

    assert_eq!(result.seed, 1234_u64);
}

#[test]
fn parse_log_level() {
    let result = parse_args_from(&argv(&["--log-level", "debug"]))
        .expect("should parse without error")
        .expect("should not trigger help/version");

    assert_eq!(result.log_level, "debug");
}

#[test]
fn parse_unknown_option_error() {
    let err =
        parse_args_from(&argv(&["--unknown"])).expect_err("unknown flag should yield an error");

    match err {
        ParseError::UnknownOption(ref flag) => {
            assert_eq!(flag, "--unknown");
        }
        other => panic!("expected UnknownOption, got {other:?}"),
    }
}

#[test]
fn parse_missing_value_error() {
    // --port without a following value.
    let err = parse_args_from(&argv(&["--port"])).expect_err("missing value should yield an error");

    match err {
        ParseError::MissingValue(ref flag) => {
            assert_eq!(flag, "--port");
        }
        other => panic!("expected MissingValue, got {other:?}"),
    }
}

#[test]
fn parse_invalid_port_error() {
    let err = parse_args_from(&argv(&["--port", "abc"]))
        .expect_err("non-numeric port should yield an error");

    match err {
        ParseError::InvalidValue {
            ref option,
            ref value,
            ..
        } => {
            assert_eq!(option, "--port");
            assert_eq!(value, "abc");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn help_flag_returns_none() {
    let result =
        parse_args_from(&argv(&["--help"])).expect("--help should not produce a ParseError");

    assert!(
        result.is_none(),
        "--help should return Ok(None) to signal the caller to exit"
    );
}

#[test]
fn version_flag_returns_none() {
    let result =
        parse_args_from(&argv(&["--version"])).expect("--version should not produce a ParseError");

    assert!(
        result.is_none(),
        "--version should return Ok(None) to signal the caller to exit"
    );
}

// ─── Additional edge-case coverage ─────────────────────────────────────────

#[test]
fn parse_all_flags_together() {
    let result = parse_args_from(&argv(&[
        "--host",
        "192.168.1.1",
        "--port",
        "9090",
        "--model",
        "/models/bonsai.gguf",
        "--tokenizer",
        "/tok/vocab.json",
        "--max-tokens",
        "512",
        "--temperature",
        "0.9",
        "--seed",
        "99",
        "--log-level",
        "warn",
    ]))
    .expect("should parse")
    .expect("should not be help/version");

    assert_eq!(result.host, "192.168.1.1");
    assert_eq!(result.port, 9090);
    assert_eq!(result.model_path, Some("/models/bonsai.gguf".to_string()));
    assert_eq!(result.tokenizer_path, Some("/tok/vocab.json".to_string()));
    assert_eq!(result.max_tokens, 512);
    assert!((result.temperature - 0.9_f32).abs() < 1e-5);
    assert_eq!(result.seed, 99);
    assert_eq!(result.log_level, "warn");
}

#[test]
fn default_server_args_matches_struct_default() {
    // Ensure the Default impl matches what the spec says. `host` is
    // "127.0.0.1" : the struct default now agrees with
    // the real, effective bind default (`BindConfig::default()`) instead of
    // the stale, never-reachable historical "0.0.0.0" literal.
    let d = ServerArgs::default();
    assert_eq!(d.host, "127.0.0.1");
    assert_eq!(d.port, 8080);
    assert_eq!(d.max_tokens, 256);
    assert_eq!(d.seed, 42);
    assert_eq!(d.log_level, "info");
    // SV-26 / SV-28: off/unset by default.
    assert!(!d.enable_ui);
    assert!(d.max_output_tokens.is_none());
}

// ─── SV-26 / SV-28: --enable-ui / --max-output-tokens ──────────────────────

#[test]
fn enable_ui_flag_is_off_by_default_and_on_when_passed() {
    let default = parse_args_from(&argv(&[]))
        .expect("should parse")
        .expect("should not be help/version");
    assert!(!default.enable_ui, "the bundled chat UI must be opt-in");

    let result = parse_args_from(&argv(&["--enable-ui"]))
        .expect("should parse")
        .expect("should not be help/version");
    assert!(result.enable_ui);
}

#[test]
fn max_output_tokens_flag_is_parsed() {
    let result = parse_args_from(&argv(&["--max-output-tokens", "4096"]))
        .expect("should parse")
        .expect("should not be help/version");
    assert_eq!(result.max_output_tokens, Some(4096));
}

#[test]
fn max_output_tokens_flag_rejects_a_non_numeric_value() {
    let err = parse_args_from(&argv(&["--max-output-tokens", "abc"]))
        .expect_err("non-numeric value should yield an error");
    match err {
        ParseError::InvalidValue { ref option, .. } => {
            assert_eq!(option, "--max-output-tokens");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn max_output_tokens_flag_rejects_zero() {
    // A zero ceiling used to parse successfully and get silently clamped to
    // 1 downstream (`RouterOptions::with_max_output_tokens_ceiling`),
    // rejecting every request outright instead of capping it -- refused at
    // parse time now, the same discipline a per-request `max_tokens: 0`
    // already gets.
    let err = parse_args_from(&argv(&["--max-output-tokens", "0"]))
        .expect_err("a zero ceiling should yield an error");
    match err {
        ParseError::InvalidValue {
            ref option,
            ref value,
            ..
        } => {
            assert_eq!(option, "--max-output-tokens");
            assert_eq!(value, "0");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn max_output_tokens_flag_missing_value_errors() {
    let err = parse_args_from(&argv(&["--max-output-tokens"]))
        .expect_err("missing value should yield an error");
    match err {
        ParseError::MissingValue(ref flag) => assert_eq!(flag, "--max-output-tokens"),
        other => panic!("expected MissingValue, got {other:?}"),
    }
}
