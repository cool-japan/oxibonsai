# Production Deployment Guide

This document covers running OxiBonsai's OpenAI-compatible HTTP server in
production: which binary to pick, TLS termination, reverse-proxy setup,
authentication, resource limits, and observability. It assumes you have
already read [`docs/CLI.md`](CLI.md) for the full flag/environment-variable
reference of both server binaries.

Everything here reflects the *actual* behavior of the code in this
repository, verified by reading the source — not aspirational config. Where
a knob doesn't exist yet, that's called out explicitly rather than implied.

## Contents

- [Which binary to run](#which-binary-to-run)
- [TLS termination](#tls-termination)
- [Reverse proxy](#reverse-proxy)
- [Authentication](#authentication)
- [Admission control (concurrency + timeouts)](#admission-control-concurrency--timeouts)
- [Rate limiting — current status](#rate-limiting--current-status)
- [CORS](#cors)
- [Resource sizing](#resource-sizing)
- [Health checks](#health-checks)
- [Prometheus metrics](#prometheus-metrics)
- [Process supervision](#process-supervision)
- [Minimal production checklist](#minimal-production-checklist)

---

## Which binary to run

OxiBonsai ships **two** independent ways to serve the OpenAI-compatible API.
They mount the same underlying router
(`oxibonsai_runtime::server::create_router_with_pool`) and therefore expose
the same HTTP surface (`/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/models`, `/admin/*`, `/health`, `/metrics`, and
`/rag/*` when RAG is enabled) — but they differ in how they're started and
supervised:

| | `oxibonsai serve` (root CLI subcommand) | `oxibonsai-serve` (standalone binary) |
|---|---|---|
| Binary | `oxibonsai` (built from `src/main.rs`, `[[bin]] name = "oxibonsai"`) | `oxibonsai-serve` (crate `oxibonsai-serve`) |
| Config surface | CLI flags + `.env` only | Layered: built-in defaults → TOML `--config` → `OXIBONSAI_*` env → CLI flags |
| Bearer auth | ✅ `--bearer-token` / env `OXIBONSAI_BEARER_TOKEN` (constant-time) | ✅ `--bearer-token` / env `OXIBONSAI_BEARER_TOKEN` / TOML `[auth].bearer_token` (constant-time) |
| Admission control (concurrency/timeout) | ✅ `--max-concurrent-requests` / `--request-timeout-ms` | ✅ `[limits].max_concurrent_requests` / `per_request_timeout_ms` |
| Graceful shutdown (SIGTERM/Ctrl-C) | ✅ calls the same `oxibonsai_runtime::server::serve_with_shutdown` helper — installs a SIGTERM/Ctrl-C handler and drains in-flight requests before exiting | ✅ `serve_with_shutdown` installs a SIGTERM/Ctrl-C handler and drains in-flight requests before exiting |
| Real client peer IP (`ConnectInfo`) | ✅ served via the same `serve_with_shutdown` helper's `into_make_service_with_connect_info::<SocketAddr>()` wiring | ✅ served via `into_make_service_with_connect_info::<SocketAddr>()` |
| Engine pool / RAG mount | ✅ `--pool-size`, `--rag` | ✅ `[limits].engine_pool_size` (config only; RAG is a separate deployment) |
| Tokenizer-kind / quantization-hint startup validation | — | ✅ `tokenizer.kind` validated against a real backend whitelist at startup; `quantization_hint` cross-checked against the loaded GGUF (non-fatal, logged) |

**Recommendation:** both binaries now shut down cleanly on SIGTERM/Ctrl-C and
see the real client peer IP, so pick based on config surface instead: use
**`oxibonsai-serve`** when you want layered TOML/env/CLI configuration and
startup validation of `tokenizer.kind` / `quantization_hint`. Use
**`oxibonsai serve`** when you want a single binary with everything
(inference + server) driven by CLI flags/`.env` only.

Both binaries are equally correct for inference — this is purely an
operational/deployment distinction, not a correctness one.

---

## TLS termination

**Neither binary terminates TLS itself.** Both bind a plain
`tokio::net::TcpListener` and serve plain HTTP via `axum::serve`; there is no
`rustls`/`native-tls` dependency anywhere in `oxibonsai-runtime` or
`oxibonsai-serve`. For any deployment reachable outside a fully trusted
private network, put a TLS-terminating reverse proxy in front and forward
plain HTTP to OxiBonsai on a loopback or private interface — e.g.
`--host 127.0.0.1` (or a private VPC address) with the proxy on `0.0.0.0:443`.

Any of the usual options work since OxiBonsai speaks plain HTTP/1.1 with SSE
streaming (`text/event-stream`) for chat completions:

- **nginx** — `proxy_pass`, with `proxy_buffering off;` and
  `proxy_read_timeout` raised well above your `--request-timeout-ms` /
  `per_request_timeout_ms` value so long generations aren't cut off by the
  proxy itself.
- **Caddy** — automatic TLS via `reverse_proxy`, no special SSE config needed
  (Caddy streams by default).
- **Cloudflare / a cloud load balancer** — terminate TLS at the edge, forward
  to the instance's private address.

Example minimal nginx config:

```nginx
server {
    listen 443 ssl;
    server_name your-host.example.com;

    ssl_certificate     /etc/letsencrypt/live/your-host.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-host.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Connection "";
        proxy_buffering off;          # required for SSE streaming responses
        proxy_read_timeout 300s;      # >= your --request-timeout-ms
    }
}
```

---

## Reverse proxy

When you front OxiBonsai with a reverse proxy, keep two things in mind:

1. **Bind OxiBonsai to a private/loopback address**, not `0.0.0.0`, so the
   inference server is only reachable through the proxy. `oxibonsai-serve`
   defaults to `0.0.0.0:8080`; `oxibonsai serve` defaults to
   `127.0.0.1:8080`. Set `--host` explicitly for your topology either way.
2. **`X-Forwarded-For`/`X-Real-IP` trust is a library-level config, not yet a
   CLI/TOML/env flag on either shipped binary.** `oxibonsai-runtime` has a
   real `RateLimitConfig::trusted_proxies: Vec<IpAddr>` allowlist (only a
   peer address in this list gets its forwarded-header claim trusted for
   per-client rate limiting), and both binaries now call the same
   `serve_with_shutdown` helper, which correctly wires real connect-info so
   that mechanism *can* see genuine peer addresses on either binary — but
   neither binary's `main.rs` currently constructs a non-default
   `RateLimitConfig` or exposes `trusted_proxies` as a flag/env/TOML key.
   See [Rate limiting](#rate-limiting--current-status) below for what this
   means in practice and how to work around it today.

---

## Authentication

Both binaries support a bearer token that gates every endpoint **except**
`/health` and `/metrics` (so load balancers and Prometheus scrapers keep
working without credentials) — this includes the mutating
`POST /admin/reset-metrics` endpoint. Comparison is constant-time (not a
plain `==`), so a shared secret doesn't leak a timing side-channel.

```bash
# oxibonsai serve
oxibonsai serve --model models/model.gguf --bearer-token "$(openssl rand -hex 32)"

# oxibonsai-serve
oxibonsai-serve --model models/model.gguf --bearer-token "$(openssl rand -hex 32)"
# or via TOML: [auth] bearer_token = "..."
# or via env:  OXIBONSAI_BEARER_TOKEN=...
```

**If you do not set a bearer token, the server is fully unauthenticated —
including `/admin/*`.** `oxibonsai serve` logs an explicit
`tracing::warn!` at startup when this happens
("all endpoints ... are unauthenticated on this listener"). Only skip
authentication when the listener is bound to `127.0.0.1` (or another
network boundary you fully trust) with no reverse-proxy exposure.

Client requests then need:

```
Authorization: Bearer <your-token>
```

---

## Admission control (concurrency + timeouts)

Both binaries bound total concurrent load with a tower admission stack —
this is separate from per-client rate limiting (see next section) and is
**on by default** with sane values:

| Setting | `oxibonsai serve` flag | `oxibonsai-serve` flag / TOML / env | Default | Behavior at the limit |
|---|---|---|---|---|
| Max concurrent requests | `--max-concurrent-requests` | `[limits].max_concurrent_requests` / `OXIBONSAI_MAX_CONCURRENT` | `32` | `503 Service Unavailable` |
| Per-request timeout | `--request-timeout-ms` | `[limits].per_request_timeout_ms` / `OXIBONSAI_REQUEST_TIMEOUT_MS` | `60000` ms | `408 Request Timeout` |

Tune `max_concurrent_requests` to roughly your `--pool-size` /
`engine_pool_size` (CPU tier) — on the GPU/Metal tier the engine pool is
always clamped to `1` replica (the Metal backend is a process-global
singleton), so raising `max_concurrent_requests` past `1` on GPU only
queues extra requests through the same single engine rather than adding
real parallelism. Raise `per_request_timeout_ms` if you serve long
`max_tokens` generations — a timeout mid-generation aborts the request with
`408` rather than truncating it gracefully.

---

## Rate limiting — current status

`oxibonsai-runtime` implements real per-client token-bucket rate limiting
(`RateLimitConfig`, `rate_limit_mw`), with the `trusted_proxies` allowlist
described above closing a header-spoofing bypass. **However, as of this
writing neither shipped binary constructs a non-default `MiddlewareConfig`**
— both build their router via `create_router_with_pool`, which passes
`MiddlewareConfig::default()` (`rate_limit: None`, rate limiting off). There
is no `--rate-limit`/`OXIBONSAI_RATE_LIMIT_*`/TOML `[rate_limit]` surface on
either binary today.

Practical implications:

- **Per-client rate limiting is not available out of the box on either
  binary.** If you need it, do one of:
  1. Rely on your reverse proxy's own rate limiting (nginx `limit_req`,
     Caddy `rate_limit`, a cloud WAF/edge rule) — the recommended default,
     since the proxy already sees real client IPs and this requires no
     OxiBonsai code changes.
  2. Embed `oxibonsai-runtime` as a library and call
     `oxibonsai_runtime::server::create_router_with_options` (or the
     equivalent) with your own `MiddlewareConfig { rate_limit: Some(RateLimitConfig { trusted_proxies: vec![your_proxy_ip], .. }), .. }`.
     Track wiring a CLI/TOML surface for this as a follow-up in `TODO.md`.
- **Admission control (above) still bounds total load** even without
  per-client rate limiting, so a single misbehaving client can't queue
  unbounded work — it can still, however, consume its fair share of the
  shared concurrency budget faster than other clients without a per-client
  cap.

---

## CORS

The default `MiddlewareConfig` both binaries use enables CORS with
`allowed_origins: ["*"]` (any origin, `GET`/`POST`/`OPTIONS`,
`Content-Type`/`Authorization` headers, no credentials). This is convenient
for local development and server-to-server API consumption, but if you
expose the API directly to browser-based clients from untrusted origins,
be aware this is currently open and — like rate limiting above — not yet
configurable via a CLI/TOML/env flag on either binary; restricting it
requires embedding `oxibonsai-runtime` as a library with a custom
`MiddlewareConfig { cors: Some(CorsConfig { allowed_origins: vec![...], .. }), .. }`.

---

## Resource sizing

Model memory is dominated by the mmap'd GGUF weights (near-zero extra RAM —
memory-mapped, not copied) plus a KV cache per **engine-pool replica**:

| Model | Format | Weights (mmap) | KV cache @ 4k context |
|---|---|---|---|
| Bonsai-8B | Q1\_0\_g128 | ~1.15 GB | ~256 MB |
| Ternary-Bonsai-8B | TQ2\_0\_g128 | ~1.75 GB | ~256 MB |
| Ternary-Bonsai-4B | TQ2\_0\_g128 | ~900 MB | ~256 MB |
| Ternary-Bonsai-1.7B | TQ2\_0\_g128 | ~390 MB | ~256 MB |

The engine pool (`--pool-size` / `[limits].engine_pool_size`, default
`min(4, CPU cores)` on the CPU tier, clamped to `1` on GPU/Metal) shares one
token-embedding table across replicas — each *additional* replica beyond
the first costs roughly one KV cache, not a full copy of the model weights.
Budget total RSS as approximately:

```
weights (mmap, ~shared, counted once)
  + pool_size × KV-cache-per-context-length
  + a modest fixed runtime overhead (metrics, tokenizer, request buffers)
```

On the GPU/Metal tier, `--pool-size` is silently clamped to `1` (the Metal
graph is a process-global singleton) — setting it higher has no effect on
GPU-tier concurrency, only admission-control queuing (see above).

For `--max-seq-len` beyond 4k, KV-cache memory scales roughly linearly with
context length; halve/double the 4k figures above accordingly.

---

## Health checks

`GET /health` returns `200 OK` unconditionally once the router is mounted —
suitable for both liveness and basic readiness probes (there is no separate
"warming up" state; by the time the router accepts connections the model
is already loaded and mmapped). It is exempt from bearer auth, so load
balancers don't need credentials.

```yaml
# Example Kubernetes probe
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

---

## Prometheus metrics

`GET /metrics` is **always** mounted (unconditionally, on both binaries) and
returns Prometheus text-exposition format
(`text/plain; version=0.0.4; charset=utf-8`), exempt from bearer auth like
`/health`. Note: `oxibonsai-serve`'s `[observability].metrics_enabled` /
`metrics_path` TOML/env fields are accepted and parsed, but as of this
writing are **not wired** to actually disable metrics or change the mount
path — `/metrics` is always live at that fixed path on both binaries. Don't
rely on `metrics_enabled = false` to hide the endpoint; put it behind your
reverse proxy's access control if you need to restrict it.

Exposed series include token throughput, prefill/decode latency,
request/error counts, per-request TBT p50/p95 + EWMA tokens/sec + queue-wait
(`oxibonsai_request_tokens_per_second`,
`oxibonsai_inter_token_latency_p50/p95_seconds`,
`oxibonsai_queue_wait_seconds`), and the KV-cache compression-level gauge
(`oxibonsai_kv_cache_compression_level` — advisory/telemetry only; see
README's "Known Limitations" for what this gauge does and doesn't do).

```yaml
# Example prometheus.yml scrape config
scrape_configs:
  - job_name: oxibonsai
    scrape_interval: 15s
    static_configs:
      - targets: ["127.0.0.1:8080"]
    metrics_path: /metrics
```

---

## Process supervision

Neither binary daemonizes itself — run it under a real process supervisor
so it restarts on crash and logs are captured centrally.

Example `systemd` unit (`oxibonsai-serve`, recommended for production per
[Which binary to run](#which-binary-to-run)):

```ini
[Unit]
Description=OxiBonsai OpenAI-compatible inference server
After=network.target

[Service]
Type=simple
User=oxibonsai
Environment=OXIBONSAI_BEARER_TOKEN=change-me-to-a-real-secret
Environment=RUST_LOG=info
ExecStart=/usr/local/bin/oxibonsai-serve --config /etc/oxibonsai/server_config.toml
Restart=on-failure
RestartSec=2
# systemd sends SIGTERM by default; oxibonsai-serve installs a SIGTERM
# handler and drains in-flight requests before exiting (see the binary
# comparison table above) — no ExecStop override needed.
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Use [`crates/oxibonsai-serve/examples/server_config.toml`](../crates/oxibonsai-serve/examples/server_config.toml)
as the canonical, fully-commented starting point for the TOML file
referenced above.

---

## Minimal production checklist

- [ ] Chose `oxibonsai-serve` (layered TOML/env config + startup validation)
      or `oxibonsai serve` (single-binary CLI convenience) — both shut down
      gracefully and wire real-peer-IP — per
      [Which binary to run](#which-binary-to-run).
- [ ] Bound to a private/loopback address; TLS terminated at a reverse
      proxy in front, not by OxiBonsai itself.
- [ ] `--bearer-token` / `OXIBONSAI_BEARER_TOKEN` set to a real secret — not
      running unauthenticated on a reachable address (`/admin/*` included).
- [ ] `--max-concurrent-requests` / `--request-timeout-ms` tuned to your
      pool size and expected `max_tokens`; reverse-proxy `proxy_read_timeout`
      (or equivalent) raised to match.
- [ ] Per-client rate limiting handled at the reverse proxy, since neither
      binary exposes it today (see [Rate limiting](#rate-limiting--current-status)).
- [ ] `/health` wired into your supervisor's liveness/readiness probe;
      `/metrics` scraped by Prometheus (both are unauthenticated by design).
- [ ] Running under a process supervisor (systemd, Docker
      `--restart=on-failure`, Kubernetes, etc.) that survives a crash.
- [ ] Resource limits sized per [Resource sizing](#resource-sizing) —
      remember the GPU/Metal tier clamps the engine pool to `1` regardless
      of `--pool-size`.
