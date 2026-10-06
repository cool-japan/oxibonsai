# oxibonsai-serve

**Status:** Stable — **Version:** 0.2.4 — **Tests:** 302 passing

Standalone OpenAI-compatible inference server for OxiBonsai.

Binary crate providing an HTTP server with `/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/v1/models`, `/health`, `/readyz` and
`/metrics`, a layered TOML / environment / flag configuration, bearer and admin
authentication, admission control, rate limiting, CORS and structured logging.
Uses hand-rolled `std::env` argument parsing — no clap dependency. Delegates the
engine and HTTP stack to [`oxibonsai-runtime`](../oxibonsai-runtime).

Part of the [OxiBonsai](https://github.com/cool-japan/oxibonsai) project.

## Usage

```sh
# Install
cargo install oxibonsai-serve

# Start a loopback-only server (the default bind is 127.0.0.1:8080)
oxibonsai-serve --model path/to/Bonsai-8B.gguf

# Reachable beyond loopback: a bearer token is mandatory (a token file keeps it out of `ps`)
oxibonsai-serve --model path/to/Bonsai-8B.gguf --host 0.0.0.0 --port 8080 \
  --bearer-token-file /etc/oxibonsai/token

# With options
oxibonsai-serve \
  --model models/Bonsai-8B.gguf \
  --max-tokens 512 \
  --temperature 0.7 \
  --log-level info
```

The server binds `127.0.0.1` by default. A non-loopback `--host` is refused at
start-up unless a bearer token is configured or `--insecure-no-auth`
acknowledges the risk. Always pass `--model`: without one the server falls back
to an untrained toy engine meant for smoke tests only.

## Probes at the concurrency limit

Admission control bounds the server to `min(max_concurrent_requests, 4 x pool
size)` requests in flight and sheds the rest with `503` + `Retry-After`. The
probe and metrics routes — `/health`, `/readyz`, `/metrics`, `/metrics/serve`, the
`observability.metrics_path` alias and `/ui/health` — are answered outside that
budget, so a liveness probe and the metrics scrape keep answering while every slot
is busy; they still honour the body limit, the rate limiter (which exempts only
`/health` and `/metrics`), bearer auth, CORS and the request timeout. `/readyz`
stays `200` at the limit (a saturated server is ready-but-saturated, and
`engine_slot_available: false` in its body says no replica is idle) and answers
`503 not_ready` only for a server that cannot serve at all. `GET /v1/models` and
every other route are inside the budget. A request
that outlasts `limits.per_request_timeout_ms` on a route with no deadline of its
own is answered `408` with `error.code: request_timeout`. See
[`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md) for the probe configuration.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config <PATH>` | none | TOML configuration file (see `examples/server_config.toml`) |
| `--model <PATH>` | none (toy fallback) | Path to GGUF model file |
| `--host <HOST>` | `127.0.0.1` | Bind address; non-loopback needs a token |
| `--port <PORT>` | `8080` | Bind port |
| `--tokenizer <PATH>` | auto | Optional tokenizer path |
| `--max-tokens <N>` | `256` | Default max tokens |
| `--temperature <F>` | `0.7` | Sampling temperature |
| `--seed <N>` | `42` | RNG seed |
| `--log-level <LEVEL>` | `info` | error/warn/info/debug/trace/off |
| `--bearer-token <TOKEN>` | optional | Bearer token, 16+ bytes (visible through `ps`) |
| `--bearer-token-file <PATH>` | optional | Read the bearer token from a file (`ps`-safe) |
| `--admin-token <TOKEN>` | optional | `/admin/*` credential (else `OXI_ADMIN_TOKEN`); `403` while unset |
| `--insecure-no-auth` | off | Allow a non-loopback host with no bearer token |
| `--cors-origin <ORIGIN>` | none (no CORS) | Allowed origin; repeatable |
| `--cors-allow-credentials` | off | `Access-Control-Allow-Credentials: true` |
| `--rate-limit-rpm <F>` | off | Per-client requests per minute |
| `--rate-limit-burst <F>` | `20` | Per-client burst |
| `--max-body-bytes <N>` | `4194304` | Request-body ceiling |
| `--enable-ui` | off | Mount the chat UI at `GET /ui` |
| `--max-output-tokens <N>` | `8192` | Ceiling on a request's `max_tokens` |

The full flag, TOML and environment-variable reference is in
[`docs/CLI.md`](../../docs/CLI.md); production guidance (TLS, reverse proxy,
limits, probes) is in [`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md).

## License

Apache-2.0 — COOLJAPAN OU
