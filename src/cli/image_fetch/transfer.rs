//! One remote image fetch, hop by hop, under the per-image deadline.
//!
//! Each hop (the URL, then each redirect target) is vetted again
//! ([`vet_target`]), gets its own client whose resolver hook answers only for
//! that hop's host ([`super::resolver::VettingResolver`]), and is sent as a
//! bare `GET`. A redirect is followed here — never by the client — at most
//! [`MAX_REMOTE_IMAGE_REDIRECTS`] times, its `Location` resolved against the
//! hop it came from and its body never read; `https` → `http` is refused. The
//! final response must be `200`; its body is streamed under the byte cap.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use oxibonsai_model::vision::remote::{
    shown_url, vet_target, RemoteFetchFailure, RemoteImageUrl, RemoteScheme, RemoteUrlRefusal,
    TargetVerdict, MAX_REMOTE_IMAGE_REDIRECTS,
};
use oxibonsai_model::vision::ImageInputError;

use super::resolver::{ResolveOutcome, ResolveRecord, VettingResolver};
use super::FetchContext;

/// `duration` in whole milliseconds.
pub(super) fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Why a fetch did not return an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FetchError {
    /// The address policy refused hop `hop` (0 is the URL itself).
    Refused {
        /// Redirects followed before the refused hop.
        hop: usize,
        /// The rule.
        refusal: RemoteUrlRefusal,
    },
    /// A permitted fetch failed at a step.
    Failed(RemoteFetchFailure),
    /// The body is larger than the cap (`bytes` is the declared length, or
    /// one past the cap when the body outgrew it).
    TooLarge {
        /// The size seen.
        bytes: usize,
        /// The cap.
        limit: usize,
    },
}

impl FetchError {
    /// The reason, for a log line (no path, query or resolved address).
    pub(crate) fn reason(&self) -> String {
        match self {
            Self::Refused { hop: 0, refusal } => refusal.to_string(),
            Self::Refused { hop, refusal } => format!("after {hop} redirect(s): {refusal}"),
            Self::Failed(failure) => failure.to_string(),
            Self::TooLarge { bytes, limit } => {
                format!("too large: {bytes} bytes against a cap of {limit}")
            }
        }
    }

    /// The typed error of the reference `url`.
    pub(crate) fn into_error(self, url: &str) -> ImageInputError {
        match self {
            Self::Refused { hop: 0, refusal } => refusal.into_error(url),
            Self::Refused { hop, refusal } => ImageInputError::RemoteUrlRefused {
                url: shown_url(url),
                reason: format!("after {hop} redirect(s): {refusal}"),
            },
            Self::Failed(failure) => failure.into_error(url),
            Self::TooLarge { bytes, limit } => ImageInputError::EncodedTooLarge { bytes, limit },
        }
    }
}

/// The outcome of one fetch, with what a log line may say about it.
#[derive(Debug)]
pub(crate) struct FetchReport {
    /// The bytes, or why not.
    pub(crate) outcome: Result<Vec<u8>, FetchError>,
    /// `scheme://host:port` of the last hop attempted.
    pub(crate) origin: String,
    /// The last status received, if any.
    pub(crate) status: Option<u16>,
    /// Redirects followed.
    pub(crate) redirects: usize,
}

impl FetchReport {
    /// A fetch that failed before reaching any server.
    pub(crate) fn failed(origin: String, failure: RemoteFetchFailure) -> Self {
        Self {
            outcome: Err(FetchError::Failed(failure)),
            origin,
            status: None,
            redirects: 0,
        }
    }
}

/// What the hops learned, kept outside the deadline's future so a timed-out
/// fetch still reports where it got to.
#[derive(Debug, Default)]
struct Trace {
    origin: String,
    status: Option<u16>,
    redirects: usize,
}

/// Fetch `url` (already parsed and vetted once by the caller) within
/// `budget` — what is left of the context's per-image deadline once the
/// fetch got its transfer slot — reading at most `max_bytes` of body. A
/// timeout names the per-image deadline itself.
pub(super) async fn fetch(
    context: &FetchContext,
    url: RemoteImageUrl,
    max_bytes: usize,
    budget: Duration,
) -> FetchReport {
    let mut trace = Trace {
        origin: url.origin(),
        ..Trace::default()
    };
    // One timer over every hop: resolution, connect, TLS, each response
    // head and the whole body. Expiry drops the future, which closes the
    // connection.
    let outcome =
        match tokio::time::timeout(budget, fetch_hops(context, url, max_bytes, &mut trace)).await {
            Ok(outcome) => outcome,
            Err(_) => Err(FetchError::Failed(RemoteFetchFailure::TimedOut {
                ms: millis(context.settings.timeout),
            })),
        };
    FetchReport {
        outcome,
        origin: trace.origin,
        status: trace.status,
        redirects: trace.redirects,
    }
}

/// The client of one hop: no redirects, no retries, no cookies, no idle
/// connection kept, the fixed `User-Agent`, the public WebPKI trust roots
/// built into the binary (the Mozilla set; not the operating system's
/// store), and a resolver hook that answers only for this hop's host.
fn hop_client(
    context: &FetchContext,
    url: &RemoteImageUrl,
    verdict: TargetVerdict,
    record: &Arc<ResolveRecord>,
) -> Result<oxihttp_client::ResolverHttpsClient, FetchError> {
    let resolver = VettingResolver {
        host: url.host().clone(),
        allowlisted: verdict == TargetVerdict::Allowlisted,
        lookup: Arc::clone(&context.lookup),
        classify: context.classify,
        record: Arc::clone(record),
    };
    oxihttp::Client::builder()
        .redirect_policy(oxihttp::RedirectPolicy::None)
        .pool_max_idle_per_host(0)
        .connect_timeout(context.settings.timeout)
        .user_agent(context.user_agent.clone())
        .with_webpki_roots()
        .with_resolver(resolver)
        .build_https_with_resolver()
        .map_err(|e| {
            FetchError::Failed(RemoteFetchFailure::Unavailable {
                reason: format!("the HTTP client could not be built: {e}"),
            })
        })
}

/// Whether the client's own URI parser reads `target` as the scheme, host
/// and port that were vetted — the guard against a second,
/// differently-behaving parse of the URL.
pub(super) fn dialled_is_vetted(target: &str, url: &RemoteImageUrl) -> bool {
    let Ok(uri) = target.parse::<oxihttp::Uri>() else {
        return false;
    };
    let host = uri
        .host()
        .map(|host| host.trim_start_matches('[').trim_end_matches(']'));
    uri.scheme_str() == Some(url.scheme().as_str())
        && host == Some(url.host().resolver_name().as_str())
        && uri.port_u16().unwrap_or(url.scheme().default_port()) == url.port()
        && uri
            .authority()
            .is_some_and(|authority| !authority.as_str().contains('@'))
}

/// The step a send error belongs to, from the hop's resolver record: a
/// policy refusal or a resolution failure the hook recorded, else a
/// connection failure (the client's `client error (Connect)`; connect and
/// TLS are one error to it), else a response failure. The connection case
/// is recognised by that text, as `oxihttp-client` 0.2.1 words it — the
/// error type carries no kind to match on; `transport_failures_name_their_step`
/// fails if an upgrade rewords it.
fn send_failure(
    error: &oxihttp::OxiHttpError,
    url: &RemoteImageUrl,
    record: &ResolveRecord,
    hop: usize,
) -> FetchError {
    match record.outcome() {
        Some(ResolveOutcome::Refused(refusal)) => FetchError::Refused { hop, refusal },
        Some(ResolveOutcome::Failed) => FetchError::Failed(RemoteFetchFailure::Resolve),
        Some(ResolveOutcome::Resolved(_)) | None => {
            if error.to_string().contains("(Connect)") {
                FetchError::Failed(match url.scheme() {
                    RemoteScheme::Https => RemoteFetchFailure::ConnectOrTls,
                    RemoteScheme::Http => RemoteFetchFailure::Connect,
                })
            } else {
                FetchError::Failed(RemoteFetchFailure::Response)
            }
        }
    }
}

/// `true` for the statuses that carry a `Location` to follow.
fn is_redirect(status: u16) -> bool {
    matches!(status, 301 | 302 | 303 | 307 | 308)
}

/// The URL a redirect from `current` leads to (hop `hop`): `location`
/// resolved against `current` under every syntax rule, and refused when it
/// would leave `https` for `http`. The next hop's address policy is applied
/// when that hop starts.
pub(super) fn redirect_target(
    current: &RemoteImageUrl,
    location: &str,
    hop: usize,
) -> Result<RemoteImageUrl, FetchError> {
    let next = current
        .join(location)
        .map_err(|refusal| FetchError::Refused { hop, refusal })?;
    if current.scheme() == RemoteScheme::Https && next.scheme() == RemoteScheme::Http {
        return Err(FetchError::Refused {
            hop,
            refusal: RemoteUrlRefusal::Downgrade,
        });
    }
    Ok(next)
}

/// Every hop of one fetch.
async fn fetch_hops(
    context: &FetchContext,
    first: RemoteImageUrl,
    max_bytes: usize,
    trace: &mut Trace,
) -> Result<Vec<u8>, FetchError> {
    let mut url = first;
    for hop in 0..=MAX_REMOTE_IMAGE_REDIRECTS {
        trace.origin = url.origin();
        let verdict = vet_target(&url, &context.settings.allowlist)
            .map_err(|refusal| FetchError::Refused { hop, refusal })?;
        let record = Arc::new(ResolveRecord::default());
        let client = hop_client(context, &url, verdict, &record)?;
        let target = url.request_target();
        if !dialled_is_vetted(&target, &url) {
            return Err(FetchError::Refused {
                hop,
                refusal: RemoteUrlRefusal::Malformed {
                    reason: "the HTTP client reads this URL differently from the address policy"
                        .to_string(),
                },
            });
        }
        let request = client.get(&target).map_err(|_| FetchError::Refused {
            hop,
            refusal: RemoteUrlRefusal::Malformed {
                reason: "the HTTP client cannot parse this URL".to_string(),
            },
        })?;
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => return Err(send_failure(&error, &url, &record, hop)),
        };
        let status = response.status().as_u16();
        trace.status = Some(status);
        if is_redirect(status) {
            if hop == MAX_REMOTE_IMAGE_REDIRECTS {
                return Err(FetchError::Failed(RemoteFetchFailure::Redirect {
                    reason: format!(
                        "more than {MAX_REMOTE_IMAGE_REDIRECTS} redirects (status {status} \
                         after {hop})"
                    ),
                }));
            }
            let location = response
                .header("location")
                .map(str::to_string)
                .ok_or_else(|| {
                    FetchError::Failed(RemoteFetchFailure::Redirect {
                        reason: format!("status {status} without a Location header"),
                    })
                })?;
            // The redirect's body is never read: the response (and its
            // connection, which the pool does not keep) goes here.
            drop(response);
            let next = redirect_target(&url, &location, hop + 1)?;
            trace.redirects += 1;
            url = next;
            continue;
        }
        if status != 200 {
            // A failure status is named; its body (which could be anything
            // the server likes) is never read or echoed.
            return Err(FetchError::Failed(RemoteFetchFailure::Status(status)));
        }
        if let Some(declared) = response.content_length() {
            let declared = usize::try_from(declared).unwrap_or(usize::MAX);
            if declared > max_bytes {
                return Err(FetchError::TooLarge {
                    bytes: declared,
                    limit: max_bytes,
                });
            }
        }
        return read_capped(response, max_bytes).await;
    }
    Err(FetchError::Failed(RemoteFetchFailure::Redirect {
        reason: format!("more than {MAX_REMOTE_IMAGE_REDIRECTS} redirects"),
    }))
}

/// Stream the body, refusing it as soon as it would pass `max_bytes`: the
/// buffer never holds more than the cap, whatever the server declared.
async fn read_capped(response: oxihttp::Response, max_bytes: usize) -> Result<Vec<u8>, FetchError> {
    use axum::body::HttpBody as _;
    let mut body = axum::body::Body::from_stream(response.body_stream());
    let mut bytes = Vec::new();
    loop {
        let frame = std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await;
        match frame {
            None => return Ok(bytes),
            Some(Err(_)) => return Err(FetchError::Failed(RemoteFetchFailure::Body)),
            Some(Ok(frame)) => {
                if let Ok(data) = frame.into_data() {
                    if bytes.len().saturating_add(data.len()) > max_bytes {
                        return Err(FetchError::TooLarge {
                            bytes: max_bytes.saturating_add(1),
                            limit: max_bytes,
                        });
                    }
                    bytes.extend_from_slice(&data);
                }
            }
        }
    }
}
