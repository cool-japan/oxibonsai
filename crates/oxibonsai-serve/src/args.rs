//! Command-line argument parsing for oxibonsai-serve.
//!
//! Pure `std::env::args()` parsing — no clap, no structopt.
//!
//! # Supported flags
//!
//! ```text
//! --config <PATH>       Path to a TOML configuration file (optional)
//! --host <HOST>         Bind host           (default: 0.0.0.0)
//! --port <PORT>         Bind port           (default: 8080)
//! --model <PATH>        Path to GGUF model file
//! --tokenizer <PATH>    Path to tokenizer   (optional)
//! --max-tokens <N>      Default max tokens  (default: 256)
//! --temperature <F>     Default temperature (default: 0.7)
//! --seed <N>            RNG seed            (default: 42)
//! --log-level <LEVEL>   Logging level       (default: info)
//! --bearer-token <TOK>  Bearer-token to require (optional)
//! --help                Print help and exit
//! --version             Print version and exit
//! ```
//!
//! The struct [`ServerArgs`] is marked `#[non_exhaustive]` so additional flags
//! can be added in a future release without breaking downstream crates
//! pattern-matching over it.

use std::collections::BTreeSet;
use std::path::PathBuf;

use thiserror::Error;

use crate::config::PartialServerConfig;

// ─── Error type ────────────────────────────────────────────────────────────

/// Errors that can occur while parsing command-line arguments.
#[derive(Debug, Error, PartialEq)]
pub enum ParseError {
    /// An unrecognised flag was encountered.
    #[error("unknown option: {0}")]
    UnknownOption(String),

    /// A flag that requires a value was given without one.
    #[error("missing value for option: {0}")]
    MissingValue(String),

    /// A value could not be interpreted as the expected type.
    #[error("invalid value '{value}' for option '{option}': {reason}")]
    InvalidValue {
        /// The option that failed to parse.
        option: String,
        /// The raw string that was rejected.
        value: String,
        /// A short explanation of why it was rejected.
        reason: String,
    },
}

// ─── ServerArgs ────────────────────────────────────────────────────────────

/// Parsed server configuration derived from argv.
///
/// `#[non_exhaustive]` so new flags can be added later without forcing a
/// major-version bump in downstream consumers.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ServerArgs {
    /// Optional TOML configuration file path.
    pub config_path: Option<String>,
    /// Host address to bind to.
    pub host: String,
    /// TCP port to listen on.
    pub port: u16,
    /// Optional path to a GGUF model file.
    pub model_path: Option<String>,
    /// Optional path to a tokenizer file/directory.
    pub tokenizer_path: Option<String>,
    /// Default maximum tokens to generate per request.
    pub max_tokens: usize,
    /// Default sampling temperature.
    pub temperature: f32,
    /// RNG seed for reproducible generation.
    pub seed: u64,
    /// Tracing log level string (e.g. "info", "debug").
    pub log_level: String,
    /// Optional bearer token to require on protected endpoints.
    pub bearer_token: Option<String>,
    /// Names of value-bearing flags that were *literally present* on the
    /// command line, as opposed to left at their built-in default.
    ///
    /// `to_partial()` consults this set (rather than comparing the parsed
    /// value against [`ServerArgs::default()`]) so that an operator who
    /// explicitly passes e.g. `--port 8080` — which happens to equal the
    /// built-in default — still overrides a non-default value set by a lower
    /// layer (TOML file / `OXIBONSAI_*` env var). Comparing against the
    /// default alone cannot distinguish "not passed" from "passed and
    /// happens to match the default", which silently breaks the documented
    /// CLI-highest-precedence guarantee for exactly that case.
    explicit_flags: BTreeSet<&'static str>,
}

impl Default for ServerArgs {
    fn default() -> Self {
        Self {
            config_path: None,
            host: "0.0.0.0".to_string(),
            port: 8080,
            model_path: None,
            tokenizer_path: None,
            max_tokens: 256,
            temperature: 0.7,
            seed: 42,
            log_level: "info".to_string(),
            bearer_token: None,
            explicit_flags: BTreeSet::new(),
        }
    }
}

impl ServerArgs {
    /// Produce a [`PartialServerConfig`] carrying only the values that the user
    /// *explicitly* specified on the command line.
    ///
    /// A flag is considered "explicit" when it was literally present in argv
    /// (tracked via `explicit_flags`, populated by [`parse_args_from`]) — not
    /// merely when its parsed value differs from [`ServerArgs::default()`].
    /// The latter would silently drop a flag such as `--port 8080` (the
    /// built-in default) when the operator passes it specifically to force
    /// the port back down from a non-default TOML/env value, breaking the
    /// documented CLI-highest-precedence layering for that case. `Option<T>`
    /// fields (`model_path`, `tokenizer_path`, `bearer_token`) have no
    /// ambiguous "default" value to collide with, so they are still checked
    /// via `.is_some()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxibonsai_serve::args::ServerArgs;
    /// let defaults = ServerArgs::default();
    /// let partial = defaults.to_partial();
    /// assert!(partial.host.is_none());
    /// assert!(partial.port.is_none());
    /// ```
    pub fn to_partial(&self) -> PartialServerConfig {
        let mut partial = PartialServerConfig::default();

        if self.explicit_flags.contains("host") {
            partial.host = Some(self.host.clone());
        }
        if self.explicit_flags.contains("port") {
            partial.port = Some(self.port);
        }
        if self.model_path.is_some() {
            partial.model_path = self.model_path.as_ref().map(PathBuf::from);
        }
        if self.tokenizer_path.is_some() {
            partial.tokenizer_path = self.tokenizer_path.as_ref().map(PathBuf::from);
        }
        if self.explicit_flags.contains("max-tokens") {
            partial.default_max_tokens = Some(self.max_tokens);
        }
        if self.explicit_flags.contains("temperature") {
            partial.default_temperature = Some(self.temperature);
        }
        if self.explicit_flags.contains("seed") {
            partial.seed = Some(self.seed);
        }
        if self.explicit_flags.contains("log-level") {
            partial.log_level = Some(self.log_level.clone());
        }
        if self.bearer_token.is_some() {
            partial.bearer_token = self.bearer_token.clone();
        }

        partial
    }
}

// ─── Parsing ───────────────────────────────────────────────────────────────

/// Parse a slice of argument strings (typically from `std::env::args().collect()`).
///
/// The first element (program name) is ignored automatically.
///
/// Returns `Ok(None)` if `--help` or `--version` was requested and handled
/// (they print to stderr/stdout and the caller should exit 0).
/// Returns `Ok(Some(args))` for a successful parse.
/// Returns `Err(ParseError)` if the arguments are malformed.
pub fn parse_args_from(argv: &[String]) -> Result<Option<ServerArgs>, ParseError> {
    let mut args = ServerArgs::default();

    // Skip argv[0] (program name) if present.
    let mut iter = argv.iter().peekable();
    // If the first token looks like a program path (no leading '--'), skip it.
    if let Some(first) = iter.peek() {
        if !first.starts_with('-') {
            iter.next();
        }
    }

    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(None);
            }
            "--version" | "-V" => {
                print_version();
                return Ok(None);
            }
            "--config" => {
                let val = next_value(&mut iter, "--config")?;
                args.config_path = Some(val.to_string());
            }
            "--host" => {
                let val = next_value(&mut iter, "--host")?;
                args.host = val.to_string();
                args.explicit_flags.insert("host");
            }
            "--port" => {
                let val = next_value(&mut iter, "--port")?;
                args.port = val.parse::<u16>().map_err(|_| ParseError::InvalidValue {
                    option: "--port".to_string(),
                    value: val.to_string(),
                    reason: "must be an integer in 1–65535".to_string(),
                })?;
                args.explicit_flags.insert("port");
            }
            "--model" => {
                let val = next_value(&mut iter, "--model")?;
                args.model_path = Some(val.to_string());
            }
            "--tokenizer" => {
                let val = next_value(&mut iter, "--tokenizer")?;
                args.tokenizer_path = Some(val.to_string());
            }
            "--max-tokens" => {
                let val = next_value(&mut iter, "--max-tokens")?;
                args.max_tokens = val.parse::<usize>().map_err(|_| ParseError::InvalidValue {
                    option: "--max-tokens".to_string(),
                    value: val.to_string(),
                    reason: "must be a non-negative integer".to_string(),
                })?;
                args.explicit_flags.insert("max-tokens");
            }
            "--temperature" => {
                let val = next_value(&mut iter, "--temperature")?;
                args.temperature = val.parse::<f32>().map_err(|_| ParseError::InvalidValue {
                    option: "--temperature".to_string(),
                    value: val.to_string(),
                    reason: "must be a floating-point number".to_string(),
                })?;
                args.explicit_flags.insert("temperature");
            }
            "--seed" => {
                let val = next_value(&mut iter, "--seed")?;
                args.seed = val.parse::<u64>().map_err(|_| ParseError::InvalidValue {
                    option: "--seed".to_string(),
                    value: val.to_string(),
                    reason: "must be a non-negative integer".to_string(),
                })?;
                args.explicit_flags.insert("seed");
            }
            "--log-level" => {
                let val = next_value(&mut iter, "--log-level")?;
                validate_log_level(val)?;
                args.log_level = val.to_string();
                args.explicit_flags.insert("log-level");
            }
            "--bearer-token" => {
                let val = next_value(&mut iter, "--bearer-token")?;
                args.bearer_token = Some(val.to_string());
            }
            other => {
                return Err(ParseError::UnknownOption(other.to_string()));
            }
        }
    }

    Ok(Some(args))
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Advance the iterator and return the next token, or return a `MissingValue`
/// error if the iterator is exhausted.
fn next_value<'a>(
    iter: &mut std::iter::Peekable<std::slice::Iter<'a, String>>,
    flag: &str,
) -> Result<&'a str, ParseError> {
    match iter.next() {
        Some(v) => Ok(v.as_str()),
        None => Err(ParseError::MissingValue(flag.to_string())),
    }
}

/// Validate that the log-level string is one of the known tracing levels.
fn validate_log_level(level: &str) -> Result<(), ParseError> {
    match level {
        "error" | "warn" | "info" | "debug" | "trace" | "off" => Ok(()),
        other => Err(ParseError::InvalidValue {
            option: "--log-level".to_string(),
            value: other.to_string(),
            reason: "must be one of: error, warn, info, debug, trace, off".to_string(),
        }),
    }
}

// ─── Help / version output ─────────────────────────────────────────────────

/// Print the help text to stderr.
pub fn print_help() {
    eprintln!(
        "\
Usage: oxibonsai-serve [OPTIONS]

Options:
  --config <PATH>       Path to a TOML configuration file (optional)
  --host <HOST>         Bind host (default: 0.0.0.0)
  --port <PORT>         Bind port (default: 8080)
  --model <PATH>        Path to GGUF model file
  --tokenizer <PATH>    Path to tokenizer (optional)
  --max-tokens <N>      Default max tokens (default: 256)
  --temperature <F>     Default temperature (default: 0.7)
  --seed <N>            RNG seed (default: 42)
  --log-level <LEVEL>   Logging level: error/warn/info/debug/trace (default: info)
  --bearer-token <TOK>  Require the given bearer token on protected endpoints
  --help, -h            Show this help
  --version, -V         Show version"
    );
}

/// Print the version string to stdout.
pub fn print_version() {
    println!("oxibonsai-serve {}", crate::banner::VERSION);
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> String {
        v.to_string()
    }

    fn args(flags: &[&str]) -> Vec<String> {
        // Prepend a fake program name so parse_args_from can skip it.
        std::iter::once("oxibonsai-serve")
            .chain(flags.iter().copied())
            .map(s)
            .collect()
    }

    #[test]
    fn defaults_are_sensible() {
        let defaults = ServerArgs::default();
        assert_eq!(defaults.port, 8080);
        assert_eq!(defaults.max_tokens, 256);
        assert!((defaults.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(defaults.seed, 42);
        assert_eq!(defaults.host, "0.0.0.0");
        assert_eq!(defaults.log_level, "info");
        assert!(defaults.model_path.is_none());
        assert!(defaults.tokenizer_path.is_none());
        assert!(defaults.config_path.is_none());
        assert!(defaults.bearer_token.is_none());
    }

    #[test]
    fn parse_empty_gives_defaults() {
        let result = parse_args_from(&args(&[])).expect("should parse");
        let parsed = result.expect("should not be help/version");
        assert_eq!(parsed, ServerArgs::default());
    }

    #[test]
    fn parse_host_port() {
        let result = parse_args_from(&args(&["--host", "127.0.0.1", "--port", "3000"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert_eq!(result.host, "127.0.0.1");
        assert_eq!(result.port, 3000);
    }

    #[test]
    fn parse_model_path() {
        let result = parse_args_from(&args(&["--model", "/path/to/model.gguf"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert_eq!(result.model_path, Some("/path/to/model.gguf".to_string()));
    }

    #[test]
    fn parse_config_path() {
        let result = parse_args_from(&args(&["--config", "/etc/oxibonsai.toml"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert_eq!(result.config_path, Some("/etc/oxibonsai.toml".to_string()));
    }

    #[test]
    fn parse_bearer_token() {
        let result = parse_args_from(&args(&["--bearer-token", "my-secret-token-abc"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert_eq!(result.bearer_token, Some("my-secret-token-abc".to_string()));
    }

    #[test]
    fn parse_temperature() {
        let result = parse_args_from(&args(&["--temperature", "0.5"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert!((result.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn parse_seed() {
        let result = parse_args_from(&args(&["--seed", "1234"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert_eq!(result.seed, 1234);
    }

    #[test]
    fn parse_log_level() {
        let result = parse_args_from(&args(&["--log-level", "debug"]))
            .expect("should parse")
            .expect("should not be help/version");
        assert_eq!(result.log_level, "debug");
    }

    #[test]
    fn parse_unknown_option_error() {
        let err = parse_args_from(&args(&["--unknown"])).expect_err("should be an error");
        assert!(matches!(err, ParseError::UnknownOption(ref s) if s == "--unknown"));
    }

    #[test]
    fn parse_missing_value_error() {
        // --port with no following value
        let err = parse_args_from(&args(&["--port"])).expect_err("should be an error");
        assert!(matches!(err, ParseError::MissingValue(ref s) if s == "--port"));
    }

    #[test]
    fn parse_invalid_port_error() {
        let err = parse_args_from(&args(&["--port", "abc"])).expect_err("should be an error");
        assert!(
            matches!(err, ParseError::InvalidValue { ref option, .. } if option == "--port"),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn help_flag_returns_none() {
        let result = parse_args_from(&args(&["--help"])).expect("should not error");
        assert!(result.is_none(), "expected None for --help");
    }

    #[test]
    fn version_flag_returns_none() {
        let result = parse_args_from(&args(&["--version"])).expect("should not error");
        assert!(result.is_none(), "expected None for --version");
    }

    #[test]
    fn to_partial_empty_for_defaults() {
        let defaults = ServerArgs::default();
        let partial = defaults.to_partial();
        assert!(partial.host.is_none());
        assert!(partial.port.is_none());
        assert!(partial.default_max_tokens.is_none());
    }

    #[test]
    fn to_partial_captures_overrides() {
        let parsed = parse_args_from(&args(&["--host", "127.0.0.1", "--port", "9000"]))
            .expect("should parse")
            .expect("should not be help/version");
        let partial = parsed.to_partial();
        assert_eq!(partial.host.as_deref(), Some("127.0.0.1"));
        assert_eq!(partial.port, Some(9000));
    }

    /// Regression test for the CLI-highest-precedence bug: passing a flag
    /// whose value happens to equal `ServerArgs::default()` must still be
    /// forwarded as an explicit `Some(..)` override, not dropped as if the
    /// flag were never passed. See finding: "ServerArgs::to_partial()
    /// silently drops CLI overrides that equal the built-in default".
    #[test]
    fn to_partial_forwards_explicit_default_valued_flags() {
        // Every one of these equals `ServerArgs::default()`'s value, but was
        // *typed* on the command line and must therefore win over a
        // lower-precedence (TOML/env) layer that set something else.
        let parsed = parse_args_from(&args(&[
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--max-tokens",
            "256",
            "--temperature",
            "0.7",
            "--seed",
            "42",
            "--log-level",
            "info",
        ]))
        .expect("should parse")
        .expect("should not be help/version");

        // Sanity: the parsed *values* really do equal the struct defaults —
        // only the (private) explicit-flags bookkeeping differs, which is
        // exactly the distinction this test is verifying `to_partial()`
        // respects.
        let defaults = ServerArgs::default();
        assert_eq!(parsed.host, defaults.host);
        assert_eq!(parsed.port, defaults.port);
        assert_eq!(parsed.max_tokens, defaults.max_tokens);
        assert!((parsed.temperature - defaults.temperature).abs() < f32::EPSILON);
        assert_eq!(parsed.seed, defaults.seed);
        assert_eq!(parsed.log_level, defaults.log_level);

        let partial = parsed.to_partial();
        assert_eq!(partial.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(partial.port, Some(8080));
        assert_eq!(partial.default_max_tokens, Some(256));
        assert_eq!(partial.default_temperature, Some(0.7));
        assert_eq!(partial.seed, Some(42));
        assert_eq!(partial.log_level.as_deref(), Some("info"));
    }

    /// The flip side of the regression above: a flag that was *not* passed
    /// at all must still merge to `None`, even though `ServerArgs` no longer
    /// derives this from comparing against `Self::default()`.
    #[test]
    fn to_partial_omits_flags_never_passed() {
        let parsed = parse_args_from(&args(&["--port", "9001"]))
            .expect("should parse")
            .expect("should not be help/version");
        let partial = parsed.to_partial();

        assert_eq!(partial.port, Some(9001));
        assert!(partial.host.is_none());
        assert!(partial.default_max_tokens.is_none());
        assert!(partial.default_temperature.is_none());
        assert!(partial.seed.is_none());
        assert!(partial.log_level.is_none());
    }
}
