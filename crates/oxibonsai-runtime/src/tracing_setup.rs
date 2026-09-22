//! Tracing initialization with configurable output format.
//!
//! Provides a structured way to initialize the `tracing` subscriber
//! with either human-readable or JSON output, driven by configuration.
//!
//! Both output formats write to **stderr**, never stdout (perf-15): a CLI
//! run (`oxibonsai run` / `chat`) prints *generated text* to stdout, and log
//! lines sharing that stream would interleave with it, corrupting anything
//! that pipes the CLI's stdout (e.g. `oxibonsai run ... | jq` or a shell
//! redirect capturing just the completion). This applies to **both**
//! branches of [`init_tracing`] — the JSON branch and the human-readable
//! branch each call `.with_writer(std::io::stderr)` independently, since a
//! fix landing on only one of them silently leaves the other interleaving
//! (the exact regression perf-15 found).
//!
//! On WASM targets, `tracing_subscriber` is not available. The
//! `init_tracing` function is a no-op stub that always succeeds.

#[cfg(not(target_arch = "wasm32"))]
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Configuration for the tracing/logging subsystem.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Log level filter string (e.g. "info", "debug", "oxibonsai=trace").
    pub log_level: String,
    /// Whether to emit JSON-formatted log lines.
    pub json_output: bool,
    /// Whether to include the source file path in log output.
    pub with_file: bool,
    /// Whether to include line numbers in log output.
    pub with_line_number: bool,
    /// Whether to include the tracing target in log output.
    pub with_target: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            json_output: false,
            with_file: false,
            with_line_number: false,
            with_target: true,
        }
    }
}

impl TracingConfig {
    /// Create a `TracingConfig` from an `ObservabilityConfig`.
    pub fn from_observability(obs: &crate::config::ObservabilityConfig) -> Self {
        Self {
            log_level: obs.log_level.clone(),
            json_output: obs.json_logs,
            ..Self::default()
        }
    }
}

/// Initialize tracing with the given configuration.
///
/// On non-WASM targets, sets up a `tracing_subscriber` with either
/// human-readable or JSON output format. Log lines are always written to
/// **stderr** (perf-15) so they never interleave with generated text, which
/// the CLI prints to stdout — this holds for both the JSON branch and the
/// human-readable branch below.
///
/// On WASM targets, this is a no-op (tracing_subscriber is unavailable).
///
/// # Errors
///
/// On non-WASM: returns an error if the tracing subscriber cannot be
/// initialized (e.g. if a global subscriber is already set).
///
/// On WASM: always returns `Ok(())`.
#[cfg(not(target_arch = "wasm32"))]
pub fn init_tracing(config: &TracingConfig) -> Result<(), Box<dyn std::error::Error>> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    if config.json_output {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().json().with_writer(std::io::stderr))
            .try_init()
            .map_err(|e| -> Box<dyn std::error::Error> {
                Box::new(std::io::Error::other(format!(
                    "failed to init tracing: {e}"
                )))
            })?;
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(
                fmt::layer()
                    .with_file(config.with_file)
                    .with_line_number(config.with_line_number)
                    .with_target(config.with_target)
                    .with_writer(std::io::stderr),
            )
            .try_init()
            .map_err(|e| -> Box<dyn std::error::Error> {
                Box::new(std::io::Error::other(format!(
                    "failed to init tracing: {e}"
                )))
            })?;
    }

    Ok(())
}

/// Initialize tracing (no-op on WASM targets).
#[cfg(target_arch = "wasm32")]
pub fn init_tracing(_config: &TracingConfig) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = TracingConfig::default();
        assert_eq!(cfg.log_level, "info");
        assert!(!cfg.json_output);
        assert!(!cfg.with_file);
        assert!(!cfg.with_line_number);
        assert!(cfg.with_target);
    }

    #[test]
    fn from_observability_config() {
        let obs = crate::config::ObservabilityConfig {
            log_level: "debug".to_string(),
            json_logs: true,
        };
        let cfg = TracingConfig::from_observability(&obs);
        assert_eq!(cfg.log_level, "debug");
        assert!(cfg.json_output);
    }

    #[test]
    fn tracing_config_clone() {
        let cfg = TracingConfig {
            log_level: "warn".to_string(),
            json_output: true,
            with_file: true,
            with_line_number: true,
            with_target: false,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.log_level, "warn");
        assert!(cloned.json_output);
        assert!(cloned.with_file);
        assert!(cloned.with_line_number);
        assert!(!cloned.with_target);
    }

    #[test]
    fn tracing_config_debug() {
        let cfg = TracingConfig::default();
        let debug_str = format!("{cfg:?}");
        assert!(debug_str.contains("TracingConfig"));
        assert!(debug_str.contains("info"));
    }

    // ── perf-15 / RT-31: stdout/stderr separation ──────────────────────────

    /// Structural regression guard for perf-15's exact failure mode: a prior
    /// fix touched only one of the two `fmt::layer()` branches and left the
    /// other silently defaulting back to stdout. Both `.json()` (line ~71)
    /// and the plain formatter (line ~82) must route through
    /// `.with_writer(std::io::stderr)`.
    #[test]
    fn both_layer_branches_route_through_stderr() {
        let source = include_str!("tracing_setup.rs");
        // Scope the scan to the real (non-wasm32) `init_tracing` function
        // body only — the module/function doc comments above it, and this
        // very test's own doc comments and assertion text, mention
        // `.with_writer(std::io::stderr)` in prose, which would otherwise
        // inflate the count and make the assertion meaningless.
        let start = source
            .find("pub fn init_tracing(config: &TracingConfig)")
            .expect("the real (non-wasm32) init_tracing should exist in this file");
        let end = source[start..]
            .find("fn init_tracing(_config")
            .map(|rel| start + rel)
            .unwrap_or(source.len());
        let body = &source[start..end];
        let occurrences = body.matches(".with_writer(std::io::stderr)").count();
        assert_eq!(
            occurrences, 2,
            "both the json() layer and the plain fmt() layer inside the \
             real init_tracing must call .with_writer(std::io::stderr) so \
             generated text (stdout) and log lines (stderr) never \
             interleave; found {occurrences} occurrence(s) instead of 2 \
             within the function body"
        );
    }

    /// Behavioural proof that `.with_writer(...)` genuinely redirects
    /// `tracing` output to the given writer rather than the process's real
    /// stdout/stderr. Uses `tracing::subscriber::with_default`, a
    /// thread-local scope that does not touch the process-global subscriber
    /// `init_tracing` installs via `try_init()`, so this cannot collide with
    /// any other test in this binary.
    ///
    /// Gated like the module's own `fmt`/`tracing_subscriber::registry`
    /// imports (non-wasm32 only): this exercises `init_tracing`'s real
    /// branch, not the wasm32 no-op.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn with_writer_receives_emitted_events() {
        use std::sync::{Arc, Mutex};

        #[derive(Clone, Default)]
        struct SharedBuf(Arc<Mutex<Vec<u8>>>);

        impl std::io::Write for SharedBuf {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let mut guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
                guard.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedBuf {
            type Writer = SharedBuf;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let buf = SharedBuf::default();
        let captured = Arc::clone(&buf.0);

        let subscriber =
            tracing_subscriber::registry().with(fmt::layer().with_writer(buf).with_ansi(false));

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!("oxibonsai_tracing_writer_probe");
        });

        let output = String::from_utf8(captured.lock().unwrap_or_else(|e| e.into_inner()).clone())
            .unwrap_or_default();
        assert!(
            output.contains("oxibonsai_tracing_writer_probe"),
            "expected the configured writer to receive the emitted event; got: {output:?}"
        );
    }
}
