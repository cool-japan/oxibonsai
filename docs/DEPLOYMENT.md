# Production Deployment Guide

This document covers running OxiBonsai's OpenAI-compatible HTTP server in
production — which binary to pick, what the safe defaults are, TLS and
reverse-proxy setup, authentication, limits, health and metrics, process
supervision — and, at the end, the release checklist the maintainers run before
publishing. It assumes you have read [`docs/CLI.md`](CLI.md) for the flag and
environment-variable reference of both server binaries.

Everything here describes the code in this repository as shipped in 0.2.4 and
was checked against the source or its tests. Where a knob does not exist, that
is said outright instead of implied, and where something could only be checked
on the maintainers' own hardware it is a release-checklist item, not a claim.

## Contents

- [Which binary to run](#which-binary-to-run)
- [Safe defaults at a glance](#safe-defaults-at-a-glance)
- [Network exposure: bind safety, TLS and reverse proxy](#network-exposure-bind-safety-tls-and-reverse-proxy)
- [Authentication](#authentication)
- [What each layer does, in order](#what-each-layer-does-in-order)
- [Admission control and timeouts](#admission-control-and-timeouts)
- [Rate limiting](#rate-limiting)
- [CORS](#cors)
- [Request and output ceilings](#request-and-output-ceilings)
- [Model integrity](#model-integrity)
- [Resource sizing](#resource-sizing)
- [Health, readiness and metrics](#health-readiness-and-metrics)
- [Process supervision and logging](#process-supervision-and-logging)
- [Security notes](#security-notes)
- [Minimal production checklist](#minimal-production-checklist)
- [Known limitations](#known-limitations)
- [Release checklist (maintainers)](#release-checklist-maintainers)

---

## Which binary to run

OxiBonsai ships **two** ways to serve the OpenAI-compatible API. They mount the
same router and expose the same HTTP surface (`/v1/chat/completions`,
`/v1/chat/completions/extended`, `/v1/completions`, `/v1/embeddings`,
`/v1/models`, `/admin/*`, `/health`, `/readyz`, `/metrics`), and both stack the
same hardening layers in the same order. They differ in how they are configured
and in what they can load.

| | `oxibonsai serve` (subcommand of the `oxibonsai` binary) | `oxibonsai-serve` (standalone binary) |
|---|---|---|
| Configuration | Flags, a flat `--config` TOML (`[server]`, `[model]`, `[sampling]`, `[observability]`), `OXIBONSAI_*` variables, `.env` | Defaults < TOML (`--config`) < `OXIBONSAI_*` environment < flags; strict start-up validation |
| Models | Everything: dense, Bonsai 2 27B hybrid, vision (`--mmproj`), `--backend`, `--ctx`, `--rope-scaling`, `--prefill-chunk`, `--ptq1-transcode` | Loads a GGUF on the auto-selected backend; no `--backend`, vision, or Bonsai 2 flags |
| Sampling defaults | The model's own `general.sampling.*` (Bonsai 2: 1.0 / 0.95 / 20), then 0.7 / 0.9 / 40 | The server's own `default_temperature` (0.7) and `default_top_p` (1.0); the GGUF is not consulted |
| Server-wide chat defaults | `--think`, `--no-think`, `--reasoning-effort`, `--tools` | none |
| Bearer token | `--bearer-token`, `--bearer-token-file`, `OXIBONSAI_BEARER_TOKEN[_FILE]` | `--bearer-token`, `--bearer-token-file`, `OXIBONSAI_BEARER_TOKEN`, `[auth]` |
| `/admin/*` token | `OXIBONSAI_ADMIN_TOKEN` / `OXI_ADMIN_TOKEN` | `--admin-token`, `[auth].admin_token`, the same variables |
| Refuses a non-loopback bind with no token | yes (`OXIBONSAI_INSECURE_NO_AUTH=1` acknowledges it) | yes (`--insecure-no-auth` acknowledges it) |
| Rate limit, CORS, body limit, UI | flags, `[server]` keys in `--config`, `OXIBONSAI_*` variables (the UI: `--enable-ui` or `[server].enable_ui`) | flags, TOML (`[rate_limit]`, `[cors]`, `[limits]`, `[ui]`) and `OXIBONSAI_*` variables |
| `metrics_enabled`, `metrics_path`, `/metrics/serve` | — | yes |
| RAG endpoints (`--rag`), `--embedding-backend` | yes | — |
| Graceful shutdown | SIGTERM and Ctrl-C drain in-flight requests (up to 30 s) | the same |
| Real peer address for rate limiting | yes | yes |
| Without a model | refuses to start | **falls back to an untrained toy `tiny_test` engine** and logs a warning |

**Recommendation.** Use **`oxibonsai serve`** for the Bonsai 2 27B, for vision
and for any Metal deployment where you want `--backend` / `--ctx` control.
Use **`oxibonsai-serve`** when you want layered TOML/environment configuration
with start-up validation for the dense models. Both are equally correct for
inference; the difference is operational. Whichever you pick, always set a model
path: the standalone binary's toy fallback exists for smoke tests and will answer
requests with nonsense.

---

## Safe defaults at a glance

A fresh install with no flags is closed by default. Every opening is explicit.

| Setting | Default | How to change it |
|---------|---------|------------------|
| Bind address | `127.0.0.1:8080` (loopback only) | `--host`, `--port`; a non-loopback host needs a bearer token or the explicit insecure acknowledgement |
| Bearer auth | off (loopback only is the assumption) | `--bearer-token-file` (preferred), `OXIBONSAI_BEARER_TOKEN` |
| `/admin/*` | `403` on every request | an admin token (`OXIBONSAI_ADMIN_TOKEN` or `OXI_ADMIN_TOKEN`) |
| CORS | no CORS headers at all | `--cors-origin`; never `*` with credentials |
| Rate limiting | off | `--rate-limit-rpm`, `--rate-limit-burst` |
| Chat UI (`/ui`) | not mounted | `--enable-ui` |
| Request body | 4 MiB, `413` above | `--max-body-bytes` |
| `max_tokens` ceiling | 8192, `400` above | `--max-output-tokens` |
| Concurrent requests | `min(32, 4 x pool size)`, `503` + `Retry-After` above (the probe and metrics routes are outside the budget) | `--max-concurrent-requests` (the ceiling is never raised above `4 x pool size`) |
| Request timeout | 60 s | `--request-timeout-ms` |
| Remote image URLs | refused (`400 image_url_fetch_disabled`), nothing opened | `--allow-image-url-fetch` (public addresses only; `--image-url-allow-host` for an intranet host); see [Security notes](#security-notes) |
| Control tokens in client text | stripped before templating | `OXI_DISABLE_PROMPT_SANITIZATION=1` (do not) |
| Model checksum | verified at start-up when a manifest lists the file | `OXIBONSAI_CHECKSUMS_FILE` |

---

## Network exposure: bind safety, TLS and reverse proxy

**Bind safety.** Both binaries refuse to start on a non-loopback `--host`
(anything other than `127.0.0.0/8`, `::1` or `localhost`) when no bearer token is
configured:

```text
refusing to bind to non-loopback host '0.0.0.0' with no bearer token configured ...
```

Either configure a token (a token read from a file counts), bind to a loopback
address and put a proxy in front, or acknowledge the risk explicitly with
`--insecure-no-auth` (standalone) / `OXIBONSAI_INSECURE_NO_AUTH=1` (both). An
admin token alone does **not** satisfy the check: it protects `/admin/*` but
leaves `/v1/*` open to the network, which is the actual danger.

**TLS.** Neither server terminates TLS. Each binds a plain TCP listener and
speaks HTTP/1.1 with SSE (`text/event-stream`) for streamed completions; there is
no TLS acceptor in either server path. (OxiBonsai's own *outbound* HTTPS, used by
`oxibonsai pull` and `oxibonsai tokenizer download`, is a separate, Pure-Rust
client.) For any deployment reachable outside a fully trusted private network,
terminate TLS in a reverse proxy and forward plain HTTP to a loopback or private
address, for example `--host 127.0.0.1` with the proxy on `:443`.

Minimal nginx:

```nginx
# In the http context (the top level of a conf.d/ file). Rate-limit here: behind a
# proxy OxiBonsai only sees the proxy's address (see "Rate limiting").
limit_req_zone $binary_remote_addr zone=oxibonsai:10m rate=2r/s;

server {
    listen 443 ssl;
    server_name your-host.example.com;

    ssl_certificate     /etc/letsencrypt/live/your-host.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-host.example.com/privkey.pem;

    location / {
        limit_req zone=oxibonsai burst=20 nodelay;
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection "";
        proxy_buffering off;          # required for SSE streaming responses
        proxy_read_timeout 600s;      # above --request-timeout-ms, and above 300 s for image workloads
    }
}
```

Caddy streams SSE by default and needs no special setting. Whichever proxy you
use, keep its read timeout above `--request-timeout-ms`, so a long generation is
cut off by OxiBonsai with a structured `504`, not by the proxy with a bare
connection reset.

---

## Authentication

A bearer token gates every endpoint **except** `/health`, `/metrics`,
`/metrics/serve` (standalone) and `/ui/health`. That includes `/readyz` and, when
mounted, `/ui` and `/rag/*`. The comparison is constant-time, and a configured
token must be at least 16 bytes.

```bash
# Prefer a file (invisible to `ps`) ...
(umask 077 && openssl rand -hex 32 > /etc/oxibonsai/token)
oxibonsai serve --model models/model.gguf --bearer-token-file /etc/oxibonsai/token
# ... or the environment:
OXIBONSAI_BEARER_TOKEN="$(cat /etc/oxibonsai/token)" oxibonsai-serve --model models/model.gguf
```

`--bearer-token <TOKEN>` works but is visible to every local user through `ps` and
lands in shell history; the server logs a warning when it is used. A token file's
contents are trimmed; an empty or unreadable file is a start-up error, never a
silent fall back to an unauthenticated server. Rotating the token means
restarting the process.

**`/admin/*` has its own credential, and the bearer token does not unlock it.**
The admin endpoints (`/admin/status`, `/admin/config`, `/admin/cache-stats`,
`/admin/workload-stats`, `POST /admin/reset-metrics`) are authenticated by an
admin token — `OXIBONSAI_ADMIN_TOKEN` or `OXI_ADMIN_TOKEN` (the standalone binary
also takes `--admin-token` and `[auth].admin_token`) — and are answered `403` for
every request while no admin token is configured, whatever the bearer token is.
Leave it unset unless you operate the server through those endpoints, and treat
the admin token like a root credential.

Clients send `Authorization: Bearer <token>`. With **no** bearer token configured
on a loopback bind the server is fully open to every local process; the start-up
log says so. That is acceptable for a developer machine and wrong for a shared
host.

---

## What each layer does, in order

Both binaries compose the router in the same order. Reading from the outside in,
which is the order a request meets them:

| # | Layer | Effect |
|---|-------|--------|
| 1 | **CORS** (only when an origin is configured) | Answers `OPTIONS` preflights before auth runs, so a browser can use an authenticated server at all. |
| 2 | **Bearer auth** (only when a token is configured) | `401` for a missing or wrong token. Exempt: `/health`, `/metrics`, `/metrics/serve`, `/ui/health`, and `OPTIONS`. |
| 3 | **Rate limit** (only when `rpm` is set) | `429` + `Retry-After`; `/health` and `/metrics` are exempt (no other route is). Sits after auth, so an unauthenticated request never consumes a token. |
| 4 | **`Content-Length` precheck** and **body limit** | A declared oversize body is a synchronous `413` before it can take a concurrency permit. |
| 5 | **Admission** | One shared concurrency budget for the whole HTTP surface, and a per-request timeout. `503` + `Retry-After` when full; a rate-limited or unauthenticated request never reaches it. The probe and metrics routes (`/health`, `/readyz`, `/metrics`, ...) are answered **outside** the budget and stay available at the limit; every other route is inside it (see [Admission control and timeouts](#admission-control-and-timeouts)). |
| 6 | **Metrics gate** (standalone) | `404` for `/metrics` when `metrics_enabled = false`. |
| 7 | **Routes** | The handlers. `/admin/*` additionally sits behind its own admin-token layer, independent of the stack above. |

Because the order is fixed, a `429` or `401` costs the server almost nothing, and
a flood of unauthenticated requests cannot starve authenticated ones.

---

## Admission control and timeouts

Admission control bounds total concurrent load and is on by default:

| Setting | `oxibonsai serve` | `oxibonsai-serve` | Default | At the limit |
|---|---|---|---|---|
| Concurrent requests | `--max-concurrent-requests` | `[limits].max_concurrent_requests`, `OXIBONSAI_MAX_CONCURRENT` | `32` | `503` + `Retry-After` |
| Request timeout | `--request-timeout-ms` | `[limits].per_request_timeout_ms`, `OXIBONSAI_REQUEST_TIMEOUT_MS` | `60000` ms | `504` (the three generation routes, and `/rag/query` when `--rag` mounts it) / `408` (other routes; see below) |

The configured concurrency is a **ceiling, not the effective limit**: the server
admits at most `4 x engine-pool size` requests at once (never fewer than 1),
unless you configured fewer. A pool of one admits four, so a burst queues briefly
and anything beyond is shed immediately with a fast `503` instead of sitting
behind one engine until the timeout fires. Size the pool first (see
[Resource sizing](#resource-sizing)), then raise `--max-concurrent-requests` only
if you want it *lower* than `4 x pool` (it cannot go higher).

**Which timeout answers.** The three generation routes —
`/v1/chat/completions`, `/v1/chat/completions/extended` and `/v1/completions` —
and, when `--rag` mounts it, `/rag/query`, carry a deadline of their own inside
the handler (the same configured value).
When it expires the in-flight generation is cancelled (and a request with
several choices or prompts starts none of the rest) and the client gets `504`
with `error.code: request_timeout`, a message naming the stage and
`error.phase` set to `preparing`, `image_fetch`, `vision_encode`,
`waiting_for_engine`, `prefill` or `decode` (with `error.generated_tokens`); a
stream that is already open carries the same error in its terminal SSE `error`
event, followed by `[DONE]`. `/rag/query` is never streamed and has no images:
it reports `preparing` while it retrieves context and tokenises the prompt,
then `waiting_for_engine`, `prefill` and `decode`. A request whose client
disconnects is cancelled the same way, streamed or not, so its replica serves
the next request instead of decoding to `max_tokens` for nobody. The admission
layer's `408` fires two seconds *later* and is a backstop for the other routes
(`/v1/embeddings`, `/v1/models`, the health and metrics probes, `/admin/*`,
`/rag/index` and `/rag/stats`); on the generation routes only a request whose
body takes longer than that grace to arrive can still get it. `/rag/index` has
no deadline of its own, but a request that is over — its client disconnected,
or the `408` ended it — stops indexing between documents and leaves the index
as it was.
The `408` uses the API's usual error envelope, with `error.type: timeout_error`
and `error.code: request_timeout` — the same code the handler's `504` carries, so
a client can treat both as "the request outlasted the server's timeout" — and,
unlike the `504`, no `error.phase`: this layer cannot know where the request was
caught. Match on `error.code`, not on the status, if you need to handle both.

**What an abandoned request still finishes.** A request abandoned (client gone, or
its deadline expired) while its images are being encoded still finishes encoding
every image it carried before the tower is free for the next request — up to 16
images, seconds each on Metal, tens of seconds each on the CPU tower; only the
generation and the remote fetches stop at once. A streamed request's SSE deadline is
counted from the moment the stream opens, so its total wall time can reach the time
spent before the stream opened plus `--request-timeout-ms` (at most twice the
setting).

**What is outside the budget.** The concurrency budget covers every route except
the probe and observability routes, which are answered outside it: `/health`,
`/readyz`, `/metrics`, and where they exist `/metrics/serve` and the
`observability.metrics_path` alias (standalone binary) and `/ui/health` (with
`--enable-ui`). They neither take nor wait for one of the `4 x pool` slots, so a
server whose slots are all busy generating still answers its liveness probe and its
metrics scrape. They are not outside anything else: the body limit, the rate
limiter, bearer auth, CORS and the request timeout apply to them exactly as to any
other route (see the table above and [Health, readiness and
metrics](#health-readiness-and-metrics)). Everything else is inside the budget —
including `GET /v1/models`, which is an API route: at the limit it, like every
generation, tokenisation, embedding and `/admin/*` request, is shed immediately
with `503`, `error.type: overloaded_error` and `Retry-After: 1`. A route added to
the server later is inside the budget unless it is named, on purpose, in the
exempt list (`ADMISSION_EXEMPT_PATHS`, in `src/cli/admission.rs` and
`crates/oxibonsai-serve/src/hardening/admission.rs`); a test enumerates the
router's routes and fails for a route that is neither exempt nor shed at the limit.

**Image requests.** With a vision projector (`oxibonsai serve --mmproj`), the vision encode and the
prefill of every image row happen inside the request's budget, on the executor the
server decodes on: on the Bonsai 2 27B the reference 256 x 192 image prompt takes
about 14 s end to end through `oxibonsai run` on the Metal hybrid runner and about
78 s on the CPU path (M3, 24 GB, loaded host); on the CPU model the image-row
prefill alone is tens of seconds to minutes (67 rows 54.9 s, 75 rows 32.6 s —
measured on separate runs at different host load; order of magnitude only — plus
about 5 s of vision encode) and grows with the image. With
`--allow-image-url-fetch`, fetching a request's remote images (one after another,
each within `--image-url-timeout-ms`) happens inside the same budget. The
server warns at start-up when `--request-timeout-ms` is below 300000 and names the
flag; set it to at least that for image workloads on the CPU model, and keep the
proxy's read timeout above it. A mismatched projector and model fail at deploy
time, not at the first image: the server refuses to start (before any weight is
bound) with `[vision_vocabulary_mismatch]` when the language model's vocabulary
does not hold `<|vision_start|>`, `<|vision_end|>` and `<|image_pad|>` at the ids
the splice uses.

Request images are decoded *before* an engine replica is leased, so their memory
is bounded separately: a source image with more pixels than 16 times what
`--image-max-tokens` can use (never below 16 megapixels, never above 8192 x 8192)
is refused as `image_too_large` from its header, before it is inflated.

**Images and the body limit.** A base64 `data:` URI is part of the JSON body, so
it counts toward `--max-body-bytes` (default: env `OXIBONSAI_MAX_BODY_BYTES`, else
4 MiB; `413` above it), and base64 inflates the image by 4/3: at the default a
data URI carries about 3 MiB of image, although the decoder accepts up to 32 MiB
encoded. Raise `--max-body-bytes` for larger inline images (about 43 MiB for a
32 MiB one) — or, with `--allow-image-url-fetch`, have clients send a large image
by URL, which the body limit does not see (the fetch is capped at 32 MiB).

---

## Rate limiting

Per-client token-bucket rate limiting is built in and **off by default**. Enable
it with `--rate-limit-rpm <N>` (`OXIBONSAI_RATE_LIMIT_RPM`; `[rate_limit].rpm` on
the standalone binary), optionally with `--rate-limit-burst <N>`
(`OXIBONSAI_RATE_LIMIT_BURST`; default `20`). A request over budget gets `429`
with `Retry-After`. `/health` and `/metrics` are never limited; no other route is
exempt, `/readyz`, `/ui/health`, `/metrics/serve` and the metrics alias included,
so point a liveness probe at `/health` and a scraper at `/metrics`. (The limiter
sits outside admission, so a `429` is the limiter's answer, never the concurrency
budget's.)

**It keys on the connecting peer's address.** Both binaries serve with real
connect-info, and the limiter trusts `X-Forwarded-For` / `X-Real-IP` only from a
configured list of trusted proxies — a list that **neither binary exposes**. So
behind a reverse proxy every request arrives from the proxy's address and shares
**one** bucket: a single noisy client can exhaust the budget for everyone. In that
topology leave the built-in limiter off (or set it as a coarse global backstop)
and rate-limit at the proxy, which already sees the real client address (nginx
`limit_req`, a Caddy rate-limit module, a cloud WAF). The built-in limiter is the
right tool when clients connect to OxiBonsai directly.

---

## CORS

CORS is **off** unless you configure an origin: with none, no CORS header is added
to any response. Configure it with `--cors-origin <ORIGIN>` (repeatable on the
standalone binary; `OXIBONSAI_CORS_ORIGIN` takes a comma-separated list; `[cors]
allowed_origins` in the standalone TOML). `--cors-allow-credentials` /
`OXIBONSAI_CORS_ALLOW_CREDENTIALS` sends `Access-Control-Allow-Credentials: true`
and requires a specific origin: a literal `*` with credentials is rejected at
start-up (the combination would let any web page read authenticated responses, and
browsers refuse it anyway). If you do not serve browser clients from another
origin, leave CORS off.

---

## Request and output ceilings

| Ceiling | Default | Over the limit |
|---------|---------|----------------|
| Request body | 4 MiB (`--max-body-bytes`) | `413` — declared size by the `Content-Length` precheck, streamed bodies by the extractor |
| Message content | 1 MiB summed over a request's messages (no flag or key) | `413 prompt_too_large`, checked before sanitising or tokenising |
| `max_tokens` / `max_completion_tokens` | 8192 (`--max-output-tokens`) | `400` naming the ceiling |
| Prompt + `max_tokens` against the context window | `--max-seq-len` (`--ctx`), or the model's own limit | `400 context_length_exceeded` naming the prompt length, the window and the room left; never a `500` from inside the engine |
| Prompt tokens (standalone) | `[limits].max_input_tokens`, 8192 | `400 max_input_tokens_exceeded`; it also sizes each replica's KV window |

`GET /v1/models` reports the window in force as `max_context_length`, so clients
can budget against it. The server rejects a non-empty `logit_bias`, a `suffix`,
`n` or `best_of` other than 1 on `/v1/completions`, and any undeclared field on
`/v1/completions`, with a `400` naming the field rather than ignoring it.

---

## Model integrity

At start-up both servers compute the model file's SHA-256 (streaming; no extra
dependency) and compare it with the checksum manifest **when the manifest lists the
file**; a mismatch refuses to start. The manifest defaults to
`scripts/checksums.sha256` **resolved relative to the current working directory**,
so it is found only when the server starts from the repository root. Set
`OXIBONSAI_CHECKSUMS_FILE=<path>` for a manifest anywhere else; no manifest, or one
that does not list the file, means no check, and the server says nothing about
that. For a deployment, either run from a directory that contains the manifest or
point the variable at an absolute path, and keep that manifest read-only.

`oxibonsai pull` is the stronger path: it verifies a named artifact fail-closed
(GGUF structure, exact size, SHA-256 against digests compiled into the binary) and
refuses a mismatch by deleting the partial download. Download with `pull`, then run
the server on the result. `oxibonsai validate --model <file>` checks that a GGUF is
well-formed before you serve it.

---

## Resource sizing

Memory is the mapped weights (shared, counted once), plus per-replica state, plus
a modest fixed overhead (metrics, tokenizer, request buffers):

| Model | Weights (file size, mmapped) |
|-------|------------------------------|
| Bonsai-8B (`Q1_0_g128`) | 1.16 GB |
| Ternary-Bonsai-8B (`TQ2_0_g128`) | 2.18 GB |
| Ternary-Bonsai-1.7B (`TQ2_0_g128`) | 540 MB |
| Ternary-Bonsai-2-27B (`PQ2_0` / `PTQ1_0` / `Q2_0` g64) | 7.21 GB / 5.95 GB / 7.63 GB |

Per-replica state:

- **Dense models.** The KV cache is `2 x layers x KV heads x head dimension x element
  size` bytes per position, times the window. The Metal runner keeps an `f16` device
  cache: 604 MB for the 8B at a 4096-token window, per replica. Memory scales
  linearly with the window, so a larger `--max-seq-len` costs real RAM.
- **Bonsai 2 27B.** The KV cache is 65 536 bytes per position (512 MiB at the default
  8192 window), plus a fixed recurrent state of about 150 MiB per sequence. A vision
  projector adds its tower: 0.87 GiB resident on the Metal tower, 1.7 GiB of `f32`
  weights on the CPU tower. The Metal tower additionally keeps a host cache of
  per-image-shape position and rotary rows — at most 8 shapes and at most 128 MiB in
  all (a shape whose rows alone exceed 128 MiB, over 6 853 merged tokens on the
  Bonsai 2 projector, is never cached) — that the 0.87 GiB figure and `oxibonsai info`
  do not count, and every image being encoded holds one more set of those rows (4 896
  bytes per patch: about 20 MB at the default 1 024-token budget, 321 MB at the
  16 384-token ceiling); budget 128 MiB plus one such set per concurrent encode on top
  of the tower's resident bytes. `oxibonsai info --model <27B.gguf> --mmproj <projector>`
  prints the plan, the largest window that fits this machine, and what is resident;
  the server refuses a `--max-seq-len` beyond the model's declared context or the
  RAM-derived bound.

**Engine pool size** (`--pool-size`, `[limits].engine_pool_size`,
`OXIBONSAI_ENGINE_POOL_SIZE`): dense CPU defaults to `min(4, CPU cores)`; a dense
Metal pool and any Bonsai 2 hybrid default to **1**. An explicit size is honoured on
the CPU and capped on Metal at `OXIBONSAI_METAL_MAX_SESSIONS` (default 4, a memory
bound); a CUDA-only build clamps to 1. Replicas share the weights and the
token-embedding table, so each extra replica costs its own KV state — but they buy
isolation and latency fairness, **not** GPU throughput (measured on an M3, 1.7B,
eight concurrent greedy requests: pools of 1, 2 and 3 served 52.2, 54.2 and 54.5
tok/s aggregate). Size the pool for how many requests you want generating at once
without queueing behind each other, not for throughput, and let admission
(`4 x pool`) shed the rest.

`OXIBONSAI_METAL_MAX_SESSIONS` is a sizing bound only: it is not enforced when a
session opens, so a Metal vision tower (a second session) and each replica count
against memory but not against it. Budget for them yourself.

---

## Health, readiness and metrics

| Endpoint | Auth | Concurrency budget | Meaning |
|----------|------|--------------------|---------|
| `GET /health` | none | **outside** | Liveness: `200` once the router is mounted. By then the model is loaded. |
| `GET /readyz` | **bearer** (when configured) | **outside** | Readiness: `200 {"status":"ready"}` while a model is loaded and the engine pool can serve — also while every replica is busy and the server is at its limit; `503 {"status":"not_ready"}` only for a server that cannot serve at all. The body always carries `model_loaded` and `engine_slot_available` (whether a replica is idle right now: information for a balancer, not part of the verdict). |
| `GET /metrics` | none | **outside** | Prometheus text exposition. |
| `GET /metrics/serve` | none | **outside** | Standalone binary only: the server's own request counters. |
| `GET /ui/health` | none | **outside** | Only with `--enable-ui`: the chat UI's own liveness ping. |
| `GET /v1/models` and every other route | bearer (when configured) | inside | `503` + `Retry-After` while the server is at its limit. |

**At the concurrency limit.** `/health`, `/readyz`, `/metrics` (and, where they
exist, `/metrics/serve`, the `observability.metrics_path` alias and `/ui/health`)
are answered outside the admission budget, so they keep answering while every one
of the `4 x pool` slots is taken. A liveness probe therefore never restarts a
server that is merely busy, and the metrics scrape does not go blind exactly when
it matters. They are *not* exempt from anything else: they still honour the body
limit, the rate limiter (which has always exempted `/health` and `/metrics` only),
bearer auth (with the exemptions listed under
[Authentication](#authentication)), CORS and the request timeout. `GET /v1/models`
is an API route and stays inside the budget:
at the limit it answers the same `503` + `Retry-After` as a generation request,
which is overload behaviour, not an outage. A server that has
`observability.metrics_enabled = false` answers `404` on its metrics routes at the
limit as at any other time.

**`/readyz` answers "can this process serve", not "is a replica idle this instant",
and a busy server is ready.** It is `200 ready` while a model is loaded and the
engine pool can hand out replicas, whatever the load: with every one of the
`4 x pool` slots taken and every replica generating (four generations on a
one-replica server) it is still `200`, with `engine_slot_available: false` in the
body to say that no replica is idle at that instant. It is `503 not_ready`
(`model_loaded: false`) only for a server that cannot serve at all. A saturated
server stays in rotation and sheds the excess on the API routes with `503` +
`Retry-After` (above) instead of reporting itself down, so the same probe is right
for a single instance and for several instances behind a balancer; a balancer
that wants least-loaded routing can read `engine_slot_available` from the body.
`/readyz` is **not** exempt from bearer auth, so a probe against a token-protected
server needs the header; use `/health`, which needs none, for liveness. (The
model's descriptor, which `/readyz` reads, is resolved at start-up, while every
replica is idle, so the probe never has to wait for a replica to answer.)

```yaml
# Kubernetes: liveness needs no credentials; readiness carries the token.
livenessProbe:
  httpGet: { path: /health, port: 8080 }
  initialDelaySeconds: 10
  periodSeconds: 10
readinessProbe:
  exec:
    command:
      - sh
      - -c
      - 'wget -q -O /dev/null --header "Authorization: Bearer $(cat /run/secrets/oxibonsai/token)" http://127.0.0.1:8080/readyz'
  periodSeconds: 5
  failureThreshold: 3
```

**Metrics.** `/metrics` is always mounted on `oxibonsai serve`; on the standalone
binary `observability.metrics_enabled = false` turns `/metrics` and `/metrics/serve`
into `404`, and `observability.metrics_path` serves the same output at an extra
path. Exposed series include `oxibonsai_requests_total`, `oxibonsai_errors_total`,
`oxibonsai_active_requests`, `oxibonsai_tokens_generated_total`,
`oxibonsai_prompt_tokens_total`, `oxibonsai_prefill_duration_seconds`,
`oxibonsai_decode_token_duration_seconds`, `oxibonsai_request_duration_seconds`,
`oxibonsai_tokens_per_second`, `oxibonsai_model_memory_bytes` and
`oxibonsai_kv_cache_utilization`. Three counters say where sampled decode on a fused
Metal engine read its logits: `oxibonsai_sampled_full_row_requests_total` (sampled
requests that decoded the full logit row — every one, by default) and
`oxibonsai_sampled_topk_steps_total` / `oxibonsai_sampled_topk_full_row_steps_total`
(steps of the GPU top-k candidate route). That route is a library opt-in
(`InferenceEngine::set_sampled_topk(SampledTopKConfig::gpu_candidates())`; neither
shipped server has a flag for it) and is off by default because its selection kernel
costs more per token than the full-row read it replaces, so on a shipped server the two
`topk` series stay 0. Four per-request rate gauges —
`oxibonsai_request_tokens_per_second`, `oxibonsai_inter_token_latency_p50_seconds`,
`oxibonsai_inter_token_latency_p95_seconds` and `oxibonsai_queue_wait_seconds` — are
fed by a library-level aggregator that neither shipped server attaches, so they
read 0 there; do not alert on them. The `oxibonsai_kv_cache_compression_level`
gauge is advisory telemetry only: the KV-cache policy computes a target level from
memory pressure but does not reconfigure the cache. The standalone binary's own
request counters (`/metrics/serve`) include the one `GET /v1/models` it sends itself
at start-up to resolve the model descriptor, so a fresh server already shows
`oxibonsai_serve_http_requests_total{route="/v1/models",status="200"} 1` before any
client has called it.

`/metrics` is unauthenticated by design so scrapers need no credentials, and it is
exempt from rate limiting. It exposes counters and timings, not prompts, but on a
reachable host restrict it at the proxy or scrape over loopback:

```yaml
scrape_configs:
  - job_name: oxibonsai
    scrape_interval: 15s
    static_configs:
      - targets: ["127.0.0.1:8080"]
    metrics_path: /metrics
```

---

## Process supervision and logging

Neither binary daemonizes itself; run it under a supervisor that restarts it on
failure. On SIGTERM or Ctrl-C both stop accepting connections and give in-flight
requests up to 30 s to finish before closing what is left, so give the supervisor
a stop timeout longer than that.

A `systemd` unit for the standalone binary, with the secrets kept out of the unit
file and the process (the token file is `0600`, owned by the service user):

```ini
[Unit]
Description=OxiBonsai OpenAI-compatible inference server
After=network.target

[Service]
Type=simple
User=oxibonsai
WorkingDirectory=/var/lib/oxibonsai
Environment=OXIBONSAI_LOG_LEVEL=info
Environment=OXIBONSAI_CHECKSUMS_FILE=/var/lib/oxibonsai/checksums.sha256
ExecStart=/usr/local/bin/oxibonsai-serve --config /etc/oxibonsai/server_config.toml \
          --bearer-token-file /etc/oxibonsai/token
Restart=on-failure
RestartSec=2
TimeoutStopSec=45
NoNewPrivileges=yes

[Install]
WantedBy=multi-user.target
```

For `oxibonsai serve`, swap the `ExecStart` for `oxibonsai serve --model ...
--bearer-token-file ... --config ...`, and set `RUST_LOG` (or `[observability]
log_level` in its `--config`) for the log level. The standalone binary does **not**
read `RUST_LOG`: its level comes from `--log-level`, `OXIBONSAI_LOG_LEVEL` or
`[observability].log_level`, and must be one of `error`, `warn`, `info`, `debug`,
`trace` or `off`. Logging goes to stdout on the standalone binary and to stderr on
`oxibonsai serve`; `[observability].json_logs = true` switches `oxibonsai serve` to
JSON lines. The `selected kernel tier` line is logged at `INFO` once per process
for each (tier, reason) pair; the further dispatchers a multi-replica pool builds
log the same fields at `DEBUG`, so four replicas show one line, not eight.

[`crates/oxibonsai-serve/examples/server_config.toml`](../crates/oxibonsai-serve/examples/server_config.toml)
is the fully commented starting point for the TOML file; a test parses and
validates it, so it cannot drift from the schema. Unknown keys in that file are
ignored without an error — check spelling against it.

---

## Security notes

- **Remote image fetching is off unless you turn it on — keep it off on a public
  endpoint unless clients need it.** A server given a vision projector accepts
  `data:` URIs and, optionally, `file://` references confined to `--media-path`
  (relative paths, no `..`, symlinks resolved and re-checked). An `http(s)`
  `image_url` is refused (`image_url_fetch_disabled`) before any connection or name
  lookup unless you pass `--allow-image-url-fetch` (or set
  `OXI_ALLOW_IMAGE_URL_FETCH=1`), because fetching URLs a client names is a
  server-side request-forgery surface. With it on:
  - **Reachable:** any public `http(s)` server a client names — the server makes a
    `GET` from its own address — plus the hosts you allowlist with
    `--image-url-allow-host <host[:port]>` / `OXI_IMAGE_URL_ALLOW_HOSTS` (use it
    for an intranet image store; an entry exempts exactly that host, and port when
    given, and nothing else).
  - **Not reachable:** loopback, private networks, carrier-grade NAT, link-local
    (the cloud metadata service at `169.254.169.254`), multicast, reserved and the
    other special-purpose IPv4 and IPv6 ranges, IPv4-mapped / NAT64 / 6to4 / Teredo
    forms, and `localhost` — in every spelling of the address, after every
    redirect (at most 3, never `https` → `http`), and whatever a host's DNS answers
    (every resolved address must be public, and the addresses vetted are the ones
    dialled, so DNS rebinding does not get through).
  - **No proxy:** the fetcher never uses `HTTP_PROXY`, `HTTPS_PROXY` or `ALL_PROXY`;
    it connects directly, so the address policy is applied to the real
    destination. Egress filtering, if you need it, belongs on the host's firewall.
  - **Bounds:** one `GET` with no cookies, credentials or retries; a `200` answer of
    at most 32 MiB; one deadline per image (`--image-url-timeout-ms`, default 10 s)
    over a wait for a transfer slot (see (e) below), resolution, connect, TLS and
    the whole body; the images of a request are
    fetched one after another and count toward `--request-timeout-ms` (a deadline
    hit mid-fetch is a `504` with `error.phase: image_fetch`). Logs record scheme,
    host, port, status, size and time — never the path or query.
  - **Settings:** `OXI_ALLOW_IMAGE_URL_FETCH`, `OXI_IMAGE_URL_TIMEOUT_MS` and
    `OXI_IMAGE_URL_ALLOW_HOSTS` are read only by a server (or `run` / `chat`)
    given `--mmproj`. There, an allowlist without the opt-in, a malformed
    allowlist entry, or (once opted in) a deadline of `0` or not a number stops
    start-up, naming the setting; `OXI_IMAGE_URL_TIMEOUT_MS` is not read without
    the opt-in. A text-only server never consults them, so a shared `.env`
    carrying them cannot stop it. The opt-in can therefore also come from a `.env`
    in a parent directory (see below); the start-up log line says whether remote
    fetching is on.
  - **What opting in still exposes.** (a) An allowlisted host is readable by every
    API client at any path, and a redirect *to* an allowlisted host is followed;
    allowlist only hosts that serve nothing but images, by `host:port`. (b) Error
    messages tell outcomes apart — a name that does not resolve, a name that
    resolves to a non-public address, a refused or timed-out connection, the HTTP
    status, and the host and scheme a redirect pointed at — so a client can probe
    which names and ports exist from the server's network position. (c) "Public"
    is decided by address class only: services that trust the server's own
    address, and internal hosts that carry globally routable IPv6 addresses, are
    reachable; restrict egress at the firewall. (d) A fetched image is not bounded
    by `--max-body-bytes`: up to 32 MiB each, 16 per request. (e) A fetch whose
    request is over — its client disconnected, or its `--request-timeout-ms`
    expired — is dropped within about 20 ms (the connection is closed; the image
    server sees it), not left to run to its own deadline, and the request starts
    no further fetch. Fetches are not counted by `--max-concurrent-requests`; the
    process bounds them itself: at most 8 transfers at once and 8 more waiting
    for one (a wait counts toward the per-image deadline), compiled in, not
    configurable by a request or a flag. A fetch beyond that is refused before
    any connection and its request answered `503` with `Retry-After: 1`,
    `error.type: overloaded_error` and `error.code: image_fetch_overloaded`.
    Each transfer holds a socket and up to 32 MiB, so the bound caps what
    request input can make the fetcher hold (about 256 MiB of bodies), but a
    client can still keep the fetcher busy for others and probe from the
    server's address within it: configure the rate limit (`--rate-limit-rpm`)
    whenever remote fetching is on.
  - **Always compiled in, inert unless opted in.** The fetcher is part of every
    native build (there is no cargo feature that removes it); without the opt-in
    no code path opens a socket or resolves a name for an image. `https` to an
    IPv6 literal (`https://[2001:db8::1]/…`) is not supported and fails closed.
- **Prompt sanitisation.** Control and special tokens (`<|im_start|>` and the like)
  are stripped from client-supplied message text before the template is assembled,
  so a client cannot forge turn boundaries. Do not set
  `OXI_DISABLE_PROMPT_SANITIZATION` on a reachable server.
- **The `.env` file.** The `oxibonsai` binary (not `oxibonsai-serve`) loads a `.env`
  from its working directory or the **first** one found in any parent directory
  (`dotenvy`) and sets each `KEY=value` as an environment variable unless that
  variable is already set. An exported variable therefore wins over the file; the
  file can carry any variable listed in the CLI reference's
  [Environment variables](CLI.md#environment-variables) — model and image paths,
  `OXIBONSAI_*` server settings, and also tokens such as `OXIBONSAI_BEARER_TOKEN`.
  A stale `.env` left in a parent directory of a deployment silently changes every
  run beneath it, and the binary does not print which file it loaded. On a server,
  prefer a supervisor-provided environment (see
  [Process supervision and logging](#process-supervision-and-logging)), keep any
  `.env` out of the directories above the working directory, and check with
  `ls -a . ..` and `env | grep '^OXI'` when behaviour surprises you.
- **Token hygiene.** Bearer and admin tokens are compared in constant time and must
  be 16 bytes or longer; a token passed as a command-line flag draws a start-up
  warning because `ps` can read it.
- **Accepted advisory `RUSTSEC-2023-0071`** (the `rsa` crate's Marvin timing
  side-channel). It sits in the Pure-Rust TLS stack `oxihttp` uses for `oxibonsai
  pull`, `oxibonsai tokenizer download` and the opt-in remote-image fetcher, with no
  avoidable edge and no upstream fix. The acceptance holds because the stack is used only as a TLS **client**
  verifying server certificates (a public-key operation), key exchange is ECDHE and
  no client certificate is configured. **It is invalidated the moment mTLS with an
  RSA client certificate is added through that stack**; `deny.toml` carries a
  greppable "INVALIDATED IF" line, and the acceptance is revisited by 2027-03-31.
  The servers themselves terminate no TLS and are not in its path.
- **Banned dependencies** (`openblas`, `bincode`, `rustfft`, `z3`, `rusqlite`, the
  C compression crates, `openssl`, `ring` and others) are enforced by `cargo deny
  check bans`; `scripts/check_pure_rust.sh` verifies the default dependency graph
  reaches no C/C++ toolchain.

---

## Minimal production checklist

- [ ] Picked the binary per [Which binary to run](#which-binary-to-run) and set an
      explicit **model path** (the standalone binary otherwise serves a toy model).
- [ ] Bound to loopback or a private address; TLS terminated at a reverse proxy in
      front. Proxy read timeout above `--request-timeout-ms`, and `proxy_buffering
      off` for SSE.
- [ ] A bearer token of 16 bytes or more, from a `0600` file or the service
      manager's environment, not from `--bearer-token`. An admin token only if you
      use `/admin/*`.
- [ ] Rate limiting at the reverse proxy (the built-in limiter shares one bucket
      behind a proxy); or the built-in limiter if clients connect directly.
- [ ] CORS left off, or set to specific origins.
- [ ] Engine pool sized deliberately; `--max-concurrent-requests` understood as a
      ceiling under `4 x pool`; `--request-timeout-ms` raised for long generations
      and image workloads.
- [ ] Context window (`--max-seq-len`) and memory budget checked with `oxibonsai
      info`; the Bonsai 2 27B plan fits the host.
- [ ] Model checksum manifest reachable (`OXIBONSAI_CHECKSUMS_FILE`, absolute path),
      or the model fetched with `oxibonsai pull`.
- [ ] `/health` for liveness; `/readyz` for readiness, with the bearer header;
      `/metrics` scraped over loopback or restricted at the proxy. All three are
      answered outside the concurrency budget, so they keep answering while the
      server is at its limit (`/readyz` stays `200`: a saturated server is
      ready-but-saturated); `GET /v1/models` and the API routes shed with `503` +
      `Retry-After` there.
- [ ] Running under a supervisor with a stop timeout above 30 s; `RUST_LOG` /
      `OXIBONSAI_LOG_LEVEL` set as appropriate for the binary.
- [ ] No stray `.env` in the working directory of the `oxibonsai` binary or any
      parent of it (it is loaded silently; see [Security notes](#security-notes)).
- [ ] A GPU build (`--features metal` or `native-cuda`) only on a host that has been
      validated for it; `native-cuda` has been run on one RTX A4000 only (Ampere,
      compute capability 8.6, CUDA 12.0, x86_64 Linux, 2026-10-07), and not on other GPU
      generations, aarch64 Linux, Windows or multi-GPU hosts.
- [ ] `--allow-image-url-fetch` left off unless clients must send images by URL;
      when on, the allowlist names only the intranet hosts that need it.

---

## Known limitations

- **Metal waits on the Bonsai 2 27B are unbounded.** The Metal hybrid runner and the
  Metal vision tower wait for each GPU command buffer without a time limit (a
  failed buffer is a typed error; a stalled one is not); only the dense batched
  prefill has the bounded wait and the cost-model router. A stalled GPU therefore
  holds the engine replica — and the tower — while the client receives its `504`,
  and a Metal engine has no CPU-tower fallback (it never loads the CPU tower).
  Restart the server if a replica stops answering.
- **No real-weight Bonsai-Image evidence in the release gate.** The `image-parity`
  capability (text-to-image on the real Bonsai-Image weights) is not required by
  any gate flag and self-skips on a host without those weights; the image
  pipeline is covered by its own parity tests and examples, not by the gate.
- **CUDA, as measured on one RTX A4000 (CUDA 12.0, x86_64 Linux, 2026-10-07).**
  The `Q4_0` / `Q8_0` / K-quant / FP8 CUDA decode uploads each weight matrix on every
  GEMV (correct, but PCIe-bound: a `Q8_0` fixture decoded at 3.8-4.0 tok/s on the GPU
  against 6.2 tok/s on the AVX-512 CPU). The Q1 batch prefill ran at 0.6-0.8x the
  per-token CUDA path on that GPU. A K-quant or FP8 model's CUDA warm-up runs a
  17-token sequential prefill (3-10 s at load). The 8B ternary model held 5739
  MiB of VRAM for a 2081 MiB file. FP8, `Q4_0`, `Q8_0` and K-quant linears decide at
  load whether they may use the GPU, so an engine forced to the reference kernel tier
  around a model loaded outside `--backend cpu` still runs their CUDA GEMVs
  (`--backend cpu` runs none of them on CUDA, nor on Metal).
  With a Q1 or ternary model, a 2-16-token prefill window after a device-KV batch
  chunk is refused (`GPU_FALLBACK_REQUIRES_CACHE_REBUILD`); `run`, `chat` and `serve`
  never emit one unless `--prefill-chunk` is 2-16 (both chunk planners fold such a
  tail into the previous window when the chunk exceeds 16 tokens, as of 0.2.4;
  verified on the CPU and on the RTX A4000 by the P14/P15 harness's `chunk=120` arm), but a library caller that drives
  `forward_prefill` with its own windows can.
- **`oxibonsai quantize` keeps the LM head F32**, so its `Q4_0` / `Q8_0` / K-quant /
  FP8 output never reaches the CUDA branches of those formats; see
  [`docs/CLI.md`](CLI.md#quantize).

---

## Release checklist (maintainers)

There is no hosted CI. Project policy allows only the `pypi-publish.yml` and
`npm-publish.yml` files in `.github/workflows/`, so the gate is a set of repository
scripts you run yourself. [`docs/ci-reference.yml`](ci-reference.yml) preserves the
retired 0.2.3 GitHub Actions workflow for anyone who wants to host the checks
elsewhere; it is a reference, weaker than `scripts/ci.sh`, and nothing runs it.

**The gate scripts**

| Script | What it does |
|--------|--------------|
| `scripts/ci.sh` | The single local gate. 21 stages, in order: `fmt`, `build-default`, `build-all-features`, `facade-image`, `facade-metal`, `facade-server-metal-image`, `clippy-all-features`, `clippy-default` (both `-D warnings`), `nextest-all-features`, `nextest-default`, `doctests`, `docs-build`, `docs-strict` (rustdoc `-D warnings`), `deny`, `pure-rust`, `cuda-syntax`, `wasm-tokenizer`, `llvm-cov`, `tmp-hardcode-advisory`, `real-model-legacy`, `real-model-bonsai2`. `--list` names the stages, `--only <stage>` runs one (refused together with `--release`, except `--only cuda-syntax`, whose run ends labelled as a partial run), `--release` makes a missing tool fail the run instead of skipping, `--accept-approximate-cuda-syntax` (only with `--release`, for a host without the CUDA toolkit) lets `cuda-syntax` pass on an approximate check and says so in the Summary, `--with-models` runs the last two real-model stages. |
| `scripts/preflight.sh` | A fast pre-push subset. `--install-hook` installs it as the git pre-push hook (`--uninstall-hook` removes it). |
| `scripts/release-gate.sh` | `ci.sh --release` plus the hardware-capability check and the real-model legs, strictly one binary at a time. Stage 0 builds the `--all-features` release CLI once and exports `OXIBONSAI_CLI_BIN` for every later stage (a value already in the environment is overridden). The real-model legs — legacy dense parity, dense embeddings, the M-18 prefill chunk sweep, speculative equals plain greedy, Metal hidden-prefill parity, the Bonsai 2 27B gates, CPU vision and Metal vision — are each required by name in the capability report, which is rotated per run so earlier evidence cannot satisfy a later run. Flags: `--require-cuda`, `--accept-approximate-cuda-syntax` (for a host without the CUDA toolkit, see the checklist below; refused together with `--require-cuda`), `--skip-legacy-models`, `--skip-bonsai2-models`, `--skip-bonsai2-metal`, `--skip-bonsai2-vision` (each skip and the CUDA waiver must be stated in the release notes), `--self-test` (262 offline scenarios against stand-in tools on macOS; 243 on Linux, where a Darwin-only block is skipped). |
| `scripts/publish.sh` | Publishes the crates in dependency order. **Dry-run by default** (`--for-real` publishes); it always runs `release-gate.sh` first, and has no `--skip-ci`. The dry run is one `cargo publish --workspace --exclude oxibonsai-testkit --dry-run --allow-dirty` — every crate packaged and verified against its siblings' local packages, since a crate-by-crate dry run cannot resolve a sibling at a version that is not on crates.io yet; the real publish goes crate by crate. It forwards `--require-cuda` and `--accept-approximate-cuda-syntax` to the gate only when you pass them. |

Two caveats about what the stages prove. `cuda-syntax` is an **approximate** check:
without `nvcc` it only parses the 31 kernel sources as C++ with CUDA builtins
stubbed, and says so; it is not a CUDA compile. With `--release` (so in the release
gate) that approximate result is **INCOMPLETE and fails the run**, because only an
`nvcc` pass counts; a host without the CUDA toolkit passes the stage only when the
owner explicitly gives `--accept-approximate-cuda-syntax` (release checklist, item
6), and the run is then labelled as waived everywhere it reports. And `ci.sh`
without `--with-models` runs **no real-model leg** — real-model parity (greedy
equality with the reference, the 27B and vision gates, throughput) comes from
`release-gate.sh` on a host that has the model files under `models/` (or
`OXIBONSAI_MODELS_DIR`).

**The coverage stage.** `llvm-cov` records a coverage baseline in
`target/coverage-baseline.txt` and sets no threshold, so it fails only when a test
fails. It runs the tests as `cargo llvm-cov nextest --workspace --profile ci`: the same
runner and `ci` profile as the `nextest-*` stages, so each test has its own process.
Plain `cargo llvm-cov` would drive `cargo test` instead, where every test of a binary
shares one instrumented process and a timing-sensitive test runs on a far slower
machine than outside coverage. The stage is hermetic with respect to real models and
says so when it starts: the variables that point a test at a model, tokenizer or weight
file (`OXI_MODEL`, `OXI_BONSAI2_*`, `OXI_REQUIRE_MODEL_FILES`, ...) are unset and
`OXIBONSAI_MODELS_DIR` names a directory that does not exist, so the test kit's
`<workspace>/models` default is never consulted. Every real-model test therefore skips
in well under a second whether or not the host has `models/`, and the coverage figures
do not depend on it. Real-model behaviour is evidenced by the `nextest-*` stages and by
the release gate's serialised real-model legs, not by coverage: under coverage counters
those tests run for hours and their timing assertions measure the instrumentation. The
stage writes its capability records to `target/coverage-capability-report.json`, never
to the release manifest `target/capability-report.json`, and fails if the release
manifest changes while it runs. A new real-model test must find its file through
`oxibonsai_testkit::workspace::{models_dir, find_model, find_model_as_named}` (or `OXIBONSAI_MODELS_DIR`),
never through a compile-time path alone, or this stage would open it. It needs both
`cargo-llvm-cov` and `cargo-nextest`, and writes one profile file per test process
under `target/llvm-cov-target` (about 10 000 files and 22 GB for the whole workspace on
the 0.2.4 tree, merged at the end and cleaned by the next run;
`cargo llvm-cov clean --workspace` reclaims the space), so check the free disk space
before running it.

**Before a release**

1. [ ] `./scripts/ci.sh --release` is green end to end, on the release commit, on a
   host with `cargo-nextest`, `cargo-llvm-cov`, `cargo-deny`, the `wasm32` target and
   the CUDA toolkit (`nvcc`) installed; without `nvcc` the `cuda-syntax` stage is
   INCOMPLETE and fails the run unless `--accept-approximate-cuda-syntax` is given
   (item 6). Record the totals in the release notes; the 0.2.4 tree's last full
   run: **9198 passed, 43 skipped** (all features, 260 test binaries) and **8487
   passed, 32 skipped** (default features).
2. [ ] `cargo deny check` is clean (bans, advisories, licenses, sources). A yanked
   crate surfaces here as an advisory failure; `cargo update -p <crate>` resolves it
   locally (`Cargo.lock` is not tracked).
3. [ ] `./scripts/release-gate.sh` is green end to end on a fresh capability report
   (`target/capability-report.json`; a skipped hardware test writes `executed: false`,
   which the gate treats as no evidence). **Run it alone on the host, under one
   external lock:** `lockf -k <lockfile> bash scripts/release-gate.sh` (macOS;
   `flock <lockfile> bash scripts/release-gate.sh` on Linux). The script runs every
   real-model leg strictly serially but has no host-wide lock of its own, so nothing
   stops a second heavy process; never run two real-model processes at once on a
   24 GB host (concurrent real-model runs have driven an 8-core/24 GB M3's load average
   past 90). On a host without the CUDA toolkit add `--accept-approximate-cuda-syntax`
   (item 6).
4. [ ] **The M-08 20 000-token YaRN run:** `OXIBONSAI_M08_RUN_LONG=1 lockf -k
   <lockfile> bash scripts/release-gate.sh` (the same external lock as step 3, and the
   same CUDA flag on a host without the toolkit). It is a
   mandatory separate step — two real decodes of roughly an hour each on
   `Bonsai-8B.gguf` — and a default run self-skips it; every release needs one passing
   run on the release commit.
5. [ ] The `wasm32` check (stage `wasm-tokenizer`; run it on its own with
   `./scripts/ci.sh --only wasm-tokenizer`): `cargo build --target
   wasm32-unknown-unknown -p oxibonsai-tokenizer`, then `RUSTFLAGS='-D warnings' cargo
   check -p oxibonsai-runtime -p oxibonsai-tokenizer --target wasm32-unknown-unknown
   --no-default-features`, which fails on any warning in a workspace crate (an unused
   import on that target, for instance).
6. [ ] `--require-cuda` only on a host with a CUDA device. Without one the CUDA
   parity checklist in `TODO.md` stays open and the release notes state the CUDA
   evidence that does exist and its scope (for 0.2.4: on one RTX A4000 on
   2026-10-07, at 7eaf006, `ci.sh --release` and `release-gate.sh --require-cuda
   --skip-bonsai2-models` passed with no waiver, and the CUDA parity harnesses ran on
   the same host; that gate requires only the `cuda` capability on Linux — its
   real-model legs are macOS-only and did not run there, and `--skip-bonsai2-models`
   means no Bonsai 2 27B evidence from that host). **On a release host without the CUDA toolkit (every macOS host) the
   `cuda-syntax` stage can never get its `nvcc` pass, so the owner must pass
   `--accept-approximate-cuda-syntax` explicitly**, to `./scripts/release-gate.sh` and
   again to `./scripts/publish.sh` (which re-runs the gate and forwards the flag only
   when it is given it). The kernel sources are then only parsed as C++ with the CUDA
   builtins stubbed; the run's verdict reads `RELEASE GATE PASSED with a waiver: CUDA
   kernel syntax was checked approximately (no nvcc)`, and its capability report
   carries a `cuda-syntax` record marked `"waived":"approximate-accepted"`. **The
   release notes must state that the CUDA backend's kernels were syntax-checked
   approximately and are not hardware-validated** by that run (name any separate
   hardware run, as above). The flag is refused together with
   `--require-cuda` (a release that requires CUDA evidence must run on a host with the
   toolkit), never waives a syntax error, and cannot waive a host with no C++ compiler
   at all, where nothing was checked. Without the flag the default stays fail-closed.
7. [ ] No source file at or above 2000 lines (`rslines 50`), and the changelog's
   `[0.2.4]` heading carries the release date.
8. [ ] `./scripts/publish.sh` (the dry run; with `--accept-approximate-cuda-syntax` on a
   host without the CUDA toolkit, item 6) passes, using an isolated **and empty**
   `CARGO_TARGET_DIR` so a concurrent build cannot race `cargo publish`'s verify step
   and no `.crate` of an earlier dry run is picked up as a dependency (observed: a
   reused directory verified a fresh `oxibonsai-cli` tarball against a stale
   `oxibonsai-runtime` tarball and failed on a symbol the old one lacked);
   only then `--for-real`, when explicitly decided.

When a gate fails, fix the cause; do not skip the gate. The only sanctioned opt-outs
are the `--skip-*` flags above and, on a host without the CUDA toolkit,
`--accept-approximate-cuda-syntax`, each of which must be named in the release notes.
