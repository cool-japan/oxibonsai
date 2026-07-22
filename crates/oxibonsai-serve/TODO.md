# oxibonsai-serve TODO

> Standalone HTTP server: argument parsing, banner, layered configuration,
> environment-variable mapping, validation, Prometheus-text metrics, and a
> bearer-auth middleware around the Axum router from `oxibonsai-runtime`.
>
> 8 source files + 7 integration test files, 178 tests (all passing).
> Version 0.2.3 — last reviewed 2026-07-21.

## Status: Stable (Alpha → Stable uplift complete)

- [x] Alpha → Stable uplift for `oxibonsai-serve`

The binary now:

- Loads a layered configuration (`defaults < TOML < env < CLI`) with
  validation baked in.
- Eagerly loads a GGUF model via `InferenceEngine::from_gguf_path` when
  `--model` is specified (and fails hard if the load fails — no silent
  fallback).
- Loads an optional tokenizer through `TokenizerBridge::from_file`.
- Applies a bearer-token authentication layer when configured, bypassing
  `/health` and `/metrics` so load balancers and Prometheus scrapers are
  unaffected.
- Ships with a canonical `examples/server_config.toml` covering every
  section.

## Done (Stable milestone)

- [x] `ServerArgs` — added `config_path`, `bearer_token`; `#[non_exhaustive]`
- [x] Pure `std::env` argument parser (zero clap/structopt dependency)
- [x] `--help` / `--version` handling
- [x] Comprehensive error messages for invalid arguments
- [x] Version banner display
- [x] Binary entry point (`main.rs`) with exit-1-on-error surface
- [x] Library interface (`lib.rs`) re-exporting args/banner/config/env/
  validation/metrics
- [x] Layered config loader (`config::ServerConfig::load`)
- [x] `OXIBONSAI_*` environment-variable parser (`env::parse_env_map`)
- [x] Validation of bounds / whitelists / file existence
  (`validation::ServerConfig::validate`)
- [x] Hand-rolled Prometheus text-exposition registry
  (`metrics::MetricsRegistry`)
- [x] Bearer-auth middleware in `main.rs`
- [x] Canonical `examples/server_config.toml`
- [x] Integration tests for config, env, validation, metrics, property-
  based invariants, and HTTP server surface (ephemeral-port booting)
- [x] **Admission control** — `tower::limit::GlobalConcurrencyLimitLayer`
  (`limits.max_concurrent_requests`) + `.timeout(limits.per_request_timeout_ms)`
  genuinely enforced via a `load_shed` + `HandleErrorLayer` stack; an
  overloaded request gets `503`, a timed-out one gets `408` (`main.rs`)
- [x] **`tokenizer.kind` backend whitelist** — rejected at validation time if
  not one of the backends this build can actually honor
  (`validation::VALID_TOKENIZER_KINDS`)
- [x] **CLI-flag-precedence fix** — a flag explicitly passed on the command
  line (even when its value equals the built-in default) now always
  outranks a lower-precedence TOML/env value, via `ServerArgs::explicit_flags`
  (`args.rs`)
- [x] **Constant-time bearer-token comparison** — local `constant_time_eq`
  helper (no `subtle` dependency) guards the bearer-auth check in `main.rs`

## Done (Alpha milestone, preserved)

- [x] `ServerArgs` — host, port, model, tokenizer, max_tokens, temperature,
  seed, log_level
- [x] Banner / version helpers
