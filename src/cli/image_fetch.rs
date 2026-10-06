//! The remote-image fetcher `run`, `chat` and `serve` install when the
//! operator opts in to remote image references (`--allow-image-url-fetch`,
//! or `OXI_ALLOW_IMAGE_URL_FETCH=1`).
//!
//! Without the opt-in nothing here runs: a remote reference is refused with
//! `image_url_fetch_disabled` by the model crate before any socket or DNS
//! query exists. With it, [`ImageUrlFetcher`] fetches each `http(s)` image
//! under the address policy of [`oxibonsai_model::vision::remote`]:
//!
//! * **URL syntax** — `http`/`https`, a host, no credentials, at most 2048
//!   bytes, no fragment sent; the dialled URL is rebuilt from the vetted
//!   parts and re-parsed by the HTTP client's own URI parser, which must
//!   agree on scheme, host and port before anything is sent
//!   ([`transfer`]).
//! * **Addresses** — an IP literal (in any spelling) is classified before any
//!   resolution, `localhost` names are refused by name, and a DNS name is
//!   resolved *inside the HTTP client's resolver hook* ([`resolver`]): the
//!   hook resolves, refuses the whole fetch if any address is not public, and
//!   hands the client only the addresses it vetted, so the addresses vetted
//!   are the addresses dialled (no "resolve, check, let the client resolve
//!   again" window for DNS rebinding).
//! * **Redirects** — never followed by the client; at most
//!   [`MAX_REMOTE_IMAGE_REDIRECTS`] are followed here, each hop vetted
//!   again, an `https` → `http` downgrade refused, a redirect's body never
//!   read.
//! * **Bounds** — `GET` with no body, no cookies, no credentials, no
//!   retries and the fixed `User-Agent: oxibonsai/<version>`; status `200`
//!   only (any other status is a typed failure naming it, its body never
//!   read); the body streamed under the encoded-image cap (a larger
//!   `Content-Length` refused before a byte is read, a missing or lying one
//!   cut off one byte past the cap); one deadline per image
//!   (`--image-url-timeout-ms`, default 10 000 ms) over name resolution,
//!   connect, TLS, the response head and the whole body, and a connect
//!   timeout no longer than it.
//! * **The operator's allowlist** (`--image-url-allow-host`,
//!   `OXI_IMAGE_URL_ALLOW_HOSTS`) exempts exactly the hosts it names from the
//!   address classes and the `localhost` rule — nothing else.
//! * **Abandonment** — a fetch made for a server's request
//!   ([`CurrentFetch`], which the runtime sets around the fetch) stops as soon
//!   as that request is abandoned (its client went away, or its deadline
//!   expired): the transfer is dropped and its connection closed within one
//!   poll interval ([`capacity::ABANDON_POLL`]), instead of running on to the
//!   per-image deadline ([`worker`]).
//! * **A process-wide bound** — at most [`capacity::MAX_FETCHES_IN_FLIGHT`]
//!   transfers run at once and [`capacity::MAX_FETCHES_WAITING`] more wait
//!   for a slot (a wait counts toward the per-image deadline); a fetch beyond
//!   both is refused at once, before any connection, and reported to its
//!   request as an overload, which a server answers with a retryable `503`
//!   (`image_fetch_overloaded`, `Retry-After`) ([`capacity`]). `run` and
//!   `chat` go through the same bound.
//!
//! # What the HTTP client does, established from its source (oxihttp 0.2.1)
//!
//! * **Proxies:** none. Neither `oxihttp`, `oxihttp-client` nor
//!   `oxihttp-core` reads an environment variable (no `env::var` in any of
//!   the three crates); a proxy is used only through `with_http_proxy` /
//!   `build_proxy*` (`oxihttp-client` `client_builder.rs:351-591`), which this
//!   module never calls, and the `HttpConnector` it builds
//!   (`client_builder.rs:967`) dials the resolved addresses directly. The
//!   tests set `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` to a listener that
//!   must see no connection.
//! * **IP-literal hosts bypass the resolver hook:** `hyper-util`'s connector
//!   parses the host with `Ipv4Addr` / `Ipv6Addr` first and dials a literal
//!   without resolving it (`hyper-util` `client/legacy/connect/http.rs:559`,
//!   `dns.rs:189`). That is why a literal is vetted here, before the client
//!   is built, and why the dialled URL carries the canonical literal (any
//!   other spelling would reach the hook as a "name").
//! * **Redirects:** followed by default (`RedirectPolicy::Limited(10)`,
//!   `redirect.rs:45`); this module sets `RedirectPolicy::None`, under which a
//!   3xx is returned as it is (`lib.rs:1124-1133`).
//! * **Retries:** none unless a `RetryPolicy` is set (`client_builder.rs:156`,
//!   `lib.rs:912-919`); none is.
//! * **Errors:** every connector failure — TCP connect and TLS handshake
//!   alike — reaches the caller as the text `client error (Connect)`
//!   (`lib.rs:1112` flattens `hyper-util`'s error, whose `Display` is only its
//!   kind, `client/legacy/client.rs:1672`). Resolution is told apart by the
//!   resolver hook's own record; an `https` connect failure is reported as
//!   "connect/TLS", because the two cannot be told apart.
//!
//! # Threads
//!
//! A fetch is called synchronously: from `run` / `chat`'s main thread (inside
//! `main`'s multi-thread runtime `block_on`, where another `block_on` would
//! panic) and from a server's blocking-pool thread
//! (`prepare_references_blocking`). Each fetcher therefore owns one
//! dedicated thread running its own current-thread runtime ([`worker`]); a
//! fetch is a job sent over a channel, and the caller waits on a plain
//! `std` channel in bounded `recv_timeout` steps, asking between them
//! whether its request was abandoned — never `block_on` on the caller's
//! thread, never a tokio primitive that panics or deadlocks there, and the
//! per-image deadline is enforced by the fetcher's own timer however busy
//! the caller's runtime is. The worker starts on the first fetch, so an
//! opted-in command that never sees a remote image never starts it.
//! Everything the HTTP client stack would log happens on that worker thread,
//! which records nothing; the fetcher's own log lines are written on the
//! calling thread and carry the scheme, host and port, the status, the byte
//! count and the milliseconds — never the path or the query, which can carry
//! credentials, and never a resolved address.

use std::net::IpAddr;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use oxibonsai_model::vision::remote::{
    classify_address, parse_allowed_host, parse_remote_image_url, vet_target, AddressClass,
    HostAllowlist, RemoteFetchFailure, DEFAULT_REMOTE_IMAGE_TIMEOUT_MS, MAX_REMOTE_IMAGE_REDIRECTS,
};
use oxibonsai_model::vision::{ImageInputError, RemoteImageFetcher, SharedRemoteImageFetcher};
use oxibonsai_runtime::vision_prefill::CurrentFetch;

mod capacity;
mod resolver;
mod transfer;
mod worker;

#[cfg(test)]
mod test_server;

use capacity::{FetchCapacity, Refusal};
pub(crate) use resolver::AddressLookup;

/// Environment fallback for `--image-url-timeout-ms`.
pub(crate) const TIMEOUT_ENV: &str = "OXI_IMAGE_URL_TIMEOUT_MS";

/// Environment fallback for `--image-url-allow-host` (comma-separated).
pub(crate) const ALLOW_HOSTS_ENV: &str = "OXI_IMAGE_URL_ALLOW_HOSTS";

/// The `User-Agent` every fetch sends.
pub(crate) fn user_agent() -> String {
    format!("oxibonsai/{}", env!("CARGO_PKG_VERSION"))
}

/// Where a setting came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettingSource {
    /// The command-line flag.
    Flag,
    /// The environment fallback.
    Env,
    /// The built-in default.
    Default,
}

impl SettingSource {
    /// How a report names it.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Flag => "flag",
            Self::Env => "environment",
            Self::Default => "default",
        }
    }
}

/// The fetch settings one command resolved: the per-image deadline and the
/// operator's allowlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ImageFetchSettings {
    /// One deadline per image (name resolution, connect, TLS, head, body).
    pub(crate) timeout: Duration,
    /// Where the deadline came from.
    pub(crate) timeout_from: SettingSource,
    /// Hosts exempt from the address classes and the `localhost` rule.
    pub(crate) allowlist: HostAllowlist,
}

impl ImageFetchSettings {
    /// Resolve the settings: the deadline from the flag, else the
    /// environment value, else [`DEFAULT_REMOTE_IMAGE_TIMEOUT_MS`]; the
    /// allowlist from the environment's comma-separated entries plus every
    /// flag entry. The environment values are passed in (not read here), so
    /// the rule is testable without touching the process environment.
    ///
    /// # Errors
    ///
    /// A deadline of `0`, a value that is not a number of milliseconds, or a
    /// malformed allowlist entry — naming the setting and the entry.
    pub(crate) fn resolve(
        timeout_flag: Option<u64>,
        timeout_env: Option<&str>,
        hosts_flag: &[String],
        hosts_env: Option<&str>,
    ) -> anyhow::Result<Self> {
        let env_timeout = timeout_env.map(str::trim).filter(|v| !v.is_empty());
        let (timeout_ms, timeout_from) = match (timeout_flag, env_timeout) {
            (Some(0), _) => anyhow::bail!(
                "--image-url-timeout-ms must be at least 1 (a number of milliseconds), got 0"
            ),
            (Some(ms), _) => (ms, SettingSource::Flag),
            (None, Some(text)) => {
                let ms: u64 = text.parse().map_err(|_| {
                    anyhow::anyhow!(
                        "{TIMEOUT_ENV}={text:?} is not a number of milliseconds (a whole number \
                         of at least 1)"
                    )
                })?;
                if ms == 0 {
                    anyhow::bail!(
                        "{TIMEOUT_ENV} must be at least 1 (a number of milliseconds), got 0"
                    );
                }
                (ms, SettingSource::Env)
            }
            (None, None) => (DEFAULT_REMOTE_IMAGE_TIMEOUT_MS, SettingSource::Default),
        };
        let mut entries = Vec::new();
        for raw in hosts_env.unwrap_or_default().split(',') {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            entries.push(parse_allowed_host(raw).map_err(|why| {
                anyhow::anyhow!("{ALLOW_HOSTS_ENV} entry {raw:?} is not a host or host:port: {why}")
            })?);
        }
        for raw in hosts_flag {
            entries.push(parse_allowed_host(raw).map_err(|why| {
                anyhow::anyhow!("--image-url-allow-host {raw:?} is not a host or host:port: {why}")
            })?);
        }
        Ok(Self {
            timeout: Duration::from_millis(timeout_ms),
            timeout_from,
            allowlist: HostAllowlist::new(entries),
        })
    }

    /// The deadline in milliseconds.
    pub(crate) fn timeout_ms(&self) -> u64 {
        u64::try_from(self.timeout.as_millis()).unwrap_or(u64::MAX)
    }

    /// One line for logs and reports: the deadline and the allowlist.
    pub(crate) fn summary(&self) -> String {
        format!(
            "per-image deadline {} ms ({}); public addresses only, plus the allowlist: {}",
            self.timeout_ms(),
            self.timeout_from.label(),
            self.allowlist
        )
    }
}

/// How a fetcher classifies an address (production: the policy's own
/// [`classify_address`]; a test can stand a loopback address in for the
/// public internet).
type Classifier = fn(IpAddr) -> Option<AddressClass>;

/// What a fetch shares with its worker.
pub(crate) struct FetchContext {
    /// The deadline and the allowlist.
    pub(crate) settings: ImageFetchSettings,
    /// How names are resolved.
    pub(crate) lookup: Arc<dyn AddressLookup>,
    /// How resolved addresses are classified.
    pub(crate) classify: Classifier,
    /// The `User-Agent` header.
    pub(crate) user_agent: String,
}

/// The remote-image fetcher (see the module docs).
pub(crate) struct ImageUrlFetcher {
    context: Arc<FetchContext>,
    worker: OnceLock<Result<worker::Worker, String>>,
    /// The transfer slots it fetches under: the process-wide bound
    /// ([`FetchCapacity::process`]) unless a test gave it its own.
    capacity: Arc<FetchCapacity>,
}

impl ImageUrlFetcher {
    /// A fetcher with `settings`, resolving names with the system resolver
    /// and classifying addresses with the address policy.
    pub(crate) fn new(settings: ImageFetchSettings) -> Self {
        Self::with_parts(settings, Arc::new(resolver::SystemLookup), classify_address)
    }

    /// A fetcher whose resolution and classification are injected — the
    /// constructor the tests stand a stub resolver (and, for the DNS
    /// rebinding case, a classifier that treats one loopback address as
    /// public) in with. It fetches under the process-wide bound.
    pub(crate) fn with_parts(
        settings: ImageFetchSettings,
        lookup: Arc<dyn AddressLookup>,
        classify: Classifier,
    ) -> Self {
        Self {
            context: Arc::new(FetchContext {
                settings,
                lookup,
                classify,
                user_agent: user_agent(),
            }),
            worker: OnceLock::new(),
            capacity: FetchCapacity::process(),
        }
    }

    /// This fetcher with transfer slots of its own instead of the
    /// process-wide bound — for tests, which run side by side in one process
    /// and must not take each other's slots.
    #[cfg(test)]
    pub(crate) fn with_capacity(mut self, capacity: Arc<FetchCapacity>) -> Self {
        self.capacity = capacity;
        self
    }

    /// The transfer slots this fetcher fetches under.
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> &Arc<FetchCapacity> {
        &self.capacity
    }

    /// This fetcher, shared, as a policy carries it.
    pub(crate) fn into_shared(self) -> SharedRemoteImageFetcher {
        SharedRemoteImageFetcher::new(Arc::new(self))
    }

    /// The worker, started on first use.
    fn worker(&self) -> Result<&worker::Worker, String> {
        self.worker
            .get_or_init(|| {
                worker::Worker::spawn(Arc::clone(&self.context))
                    .map_err(|e| format!("its fetch thread could not start: {e}"))
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

impl RemoteImageFetcher for ImageUrlFetcher {
    fn fetch(&self, url: &str, max_bytes: usize) -> Result<Vec<u8>, ImageInputError> {
        let started = Instant::now();
        let parsed = parse_remote_image_url(url).map_err(|refusal| refusal.into_error(url))?;
        // Everything that needs no resolver is decided here, before a job
        // (or the worker thread itself) exists.
        if let Err(refusal) = vet_target(&parsed, &self.context.settings.allowlist) {
            tracing::warn!(
                origin = %parsed.origin(),
                reason = %refusal,
                "remote image refused before any connection"
            );
            return Err(refusal.into_error(url));
        }
        // The request this fetch is for, when a server's request asked for
        // it: the fetch stops as soon as that request is abandoned.
        let request = CurrentFetch::on_this_thread();
        let abandoned = || request.as_ref().is_some_and(CurrentFetch::is_abandoned);
        if abandoned() {
            return Err(RemoteFetchFailure::Abandoned.into_error(url));
        }
        let worker = self
            .worker()
            .map_err(|reason| RemoteFetchFailure::Unavailable { reason }.into_error(url))?;
        let timeout = self.context.settings.timeout;
        let deadline = worker::FetchDeadline {
            at: started + timeout,
            timeout,
        };
        let permit = match self.capacity.admit(&abandoned, deadline.at) {
            Ok(permit) => permit,
            Err(refusal) => return Err(self.refused(&parsed, url, refusal, request.as_ref())),
        };
        let report = worker.run(parsed, max_bytes, deadline, permit, &abandoned);
        let millis = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        match &report.outcome {
            Ok(bytes) => tracing::info!(
                origin = %report.origin,
                status = report.status.unwrap_or_default(),
                bytes = bytes.len(),
                redirects = report.redirects,
                ms = millis,
                "remote image fetched"
            ),
            Err(error) => tracing::warn!(
                origin = %report.origin,
                status = report.status.unwrap_or_default(),
                redirects = report.redirects,
                ms = millis,
                reason = %error.reason(),
                "remote image not fetched"
            ),
        }
        report.outcome.map_err(|error| error.into_error(url))
    }

    fn preflight(&self, url: &str) -> Result<(), ImageInputError> {
        let parsed = parse_remote_image_url(url).map_err(|refusal| refusal.into_error(url))?;
        vet_target(&parsed, &self.context.settings.allowlist)
            .map(drop)
            .map_err(|refusal| refusal.into_error(url))
    }

    fn describe(&self) -> String {
        format!(
            "remote image fetcher ({}; at most {MAX_REMOTE_IMAGE_REDIRECTS} redirects; at most \
             {} fetches in flight and {} waiting)",
            self.context.settings.summary(),
            self.capacity.in_flight_limit(),
            self.capacity.waiting_limit()
        )
    }
}

impl ImageUrlFetcher {
    /// The error of a fetch that got no transfer slot (nothing was opened):
    /// for a fetcher at capacity, the overload is reported to the request
    /// (a server answers it as a retryable `503`) and the failure names the
    /// bound; a fetch whose request was abandoned, or whose deadline passed,
    /// while it waited fails as such.
    fn refused(
        &self,
        parsed: &oxibonsai_model::vision::remote::RemoteImageUrl,
        url: &str,
        refusal: Refusal,
        request: Option<&CurrentFetch>,
    ) -> ImageInputError {
        let failure = match refusal {
            Refusal::Overloaded => {
                let counts = self.capacity.counts();
                tracing::warn!(
                    origin = %parsed.origin(),
                    in_flight = counts.in_flight,
                    waiting = counts.waiting,
                    "remote image refused: the fetcher is at capacity"
                );
                if let Some(request) = request {
                    request.report_overloaded();
                }
                RemoteFetchFailure::Unavailable {
                    reason: format!(
                        "it is at capacity ({} fetches in flight and {} waiting, the most this \
                         process runs at once); retry after a short backoff",
                        self.capacity.in_flight_limit(),
                        self.capacity.waiting_limit()
                    ),
                }
            }
            Refusal::Abandoned => {
                tracing::info!(
                    origin = %parsed.origin(),
                    "remote image not fetched: its request was abandoned while it waited"
                );
                RemoteFetchFailure::Abandoned
            }
            Refusal::TimedOut => {
                let ms =
                    u64::try_from(self.context.settings.timeout.as_millis()).unwrap_or(u64::MAX);
                tracing::warn!(
                    origin = %parsed.origin(),
                    ms,
                    "remote image not fetched: no transfer slot freed up within its deadline"
                );
                RemoteFetchFailure::TimedOut { ms }
            }
        };
        failure.into_error(url)
    }
}

/// What a command reports about remote images at start-up (`serve`'s log,
/// `info`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RemoteFetchReport {
    /// No opt-in: remote references are refused.
    Disabled,
    /// Opted in (by the flag, or the environment), with these settings.
    Enabled {
        /// Where the opt-in came from.
        from: SettingSource,
        /// The resolved settings.
        settings: ImageFetchSettings,
    },
}

impl std::fmt::Display for RemoteFetchReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disabled => f.write_str(
                "remote image URLs are refused (image_url_fetch_disabled); opt in with \
                 --allow-image-url-fetch or OXI_ALLOW_IMAGE_URL_FETCH=1",
            ),
            Self::Enabled { from, settings } => write!(
                f,
                "remote image URLs are fetched (opted in by the {}): {}; at most \
                 {MAX_REMOTE_IMAGE_REDIRECTS} redirects, no proxy",
                from.label(),
                settings.summary()
            ),
        }
    }
}

#[cfg(test)]
#[path = "image_fetch_tests.rs"]
mod tests;

#[cfg(test)]
mod abandon_tests;

#[cfg(all(test, feature = "server"))]
mod serve_tests;
