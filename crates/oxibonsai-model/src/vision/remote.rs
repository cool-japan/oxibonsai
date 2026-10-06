//! Remote (`http` / `https`) image references: the operator's opt-in, the
//! seam a fetcher plugs into, and the pure address policy every fetcher
//! applies.
//!
//! Nothing here opens a socket or resolves a name: the module is the part of
//! remote-image handling that needs no I/O, so a library user who writes
//! their own fetcher reuses exactly the rules the `oxibonsai` command applies.
//!
//! # Three states
//!
//! [`ImageSourcePolicy::remote`](super::ImageSourcePolicy::remote) is a
//! [`RemoteImageAccess`]:
//!
//! | state | what an `http(s)` reference gets |
//! |---|---|
//! | [`RemoteImageAccess::Disabled`] (the default) | `image_url_fetch_disabled`; no connection, no DNS query |
//! | [`RemoteImageAccess::OptedInWithoutFetcher`] | `image_url_fetch_disabled`, saying the opt-in was given but the application installed no fetcher |
//! | [`RemoteImageAccess::Fetcher`] | fetched through the installed [`RemoteImageFetcher`] |
//!
//! Opting in is the operator's decision (`--allow-image-url-fetch`, or
//! `OXI_ALLOW_IMAGE_URL_FETCH=1`); installing a fetcher is the front end's
//! (the `oxibonsai` command installs one with this module's address policy
//! when the operator opted in).
//!
//! # URL syntax ([`parse_remote_image_url`])
//!
//! Only `http` and `https` (any case), with a host, without credentials
//! (`user:password@` is refused), at most [`MAX_REMOTE_IMAGE_URL_BYTES`]
//! bytes, no whitespace, control character or backslash anywhere (a
//! backslash is where URL parsers disagree about the host). The fragment is
//! dropped and never sent. A host that ends in a number is an IPv4 literal
//! and is read the way browsers and the C library's `inet_aton` read one —
//! `2130706433`, `0x7f.0.0.1`, `0177.0.0.1` and `127.1` are all `127.0.0.1`
//! — so no spelling of an address reaches a resolver as a name; an IPv6
//! literal must be bracketed and may not carry a zone identifier; anything
//! else is a lower-cased DNS name (one trailing dot removed). The parsed
//! [`RemoteImageUrl`] is what a fetcher dials: its
//! [`RemoteImageUrl::request_target`] is rebuilt from the vetted parts, so the
//! host that was vetted is the host that is dialled, never a second,
//! differently-behaving parse of the original text.
//!
//! # Address policy ([`classify_address`])
//!
//! An address is fetched from only when it is a public unicast address.
//! Refused, IPv4: `0.0.0.0/8`, `10/8`, `100.64/10`, `127/8`, `169.254/16`
//! (cloud metadata lives at `169.254.169.254`), `172.16/12`, `192.0.0.0/24`,
//! `192.0.2.0/24`, `192.88.99.0/24`, `192.168/16`, `198.18/15`,
//! `198.51.100.0/24`, `203.0.113.0/24`, `224/4`, `240/4` and
//! `255.255.255.255`. IPv6: `::`, `::1`, every IPv4-mapped (`::ffff:0:0/96`)
//! and IPv4-compatible (`::/96`) address whatever IPv4 address it embeds,
//! NAT64 (`64:ff9b::/96`, `64:ff9b:1::/48`), discard-only `100::/64`, the
//! IETF protocol assignments `2001::/23` (Teredo `2001::/32` and
//! benchmarking `2001:2::/48` among them), documentation (`2001:db8::/32`,
//! `3fff::/20`), 6to4 `2002::/16`, unique-local `fc00::/7`, link-local
//! `fe80::/10`, site-local `fec0::/10`, multicast `ff00::/8`, and everything
//! outside the global-unicast `2000::/3`.
//!
//! A host is fetched only when **every** address it resolves to is public
//! ([`vet_resolved`]): one refused address refuses the whole fetch (a
//! fetcher never "tries the next record"). `localhost` and `*.localhost` are
//! refused by name before any resolution ([`vet_target`]).
//!
//! # The operator's allowlist ([`HostAllowlist`])
//!
//! `host[:port]` entries (`--image-url-allow-host`, `OXI_IMAGE_URL_ALLOW_HOSTS`)
//! match a URL's host exactly and case-insensitively — an IP literal by its
//! address, whatever spelling either side used — and its port when the entry
//! names one. A matching host is exempt from the address classes and from
//! the `localhost` name rule, and from nothing else: the scheme, credential,
//! redirect, size and deadline rules still apply, and every redirect hop
//! must itself be allowlisted or public.
//!
//! # Redirects ([`RemoteImageUrl::join`])
//!
//! A fetcher follows at most [`MAX_REMOTE_IMAGE_REDIRECTS`] redirects
//! itself; each `Location` is resolved against the URL it came from
//! (RFC 3986 §5.2, dot segments removed) into a fresh [`RemoteImageUrl`]
//! that goes through every rule above again; an `https` → `http` downgrade
//! is refused ([`RemoteUrlRefusal::Downgrade`]).

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Arc;

use super::image_decode::ImageInputError;

/// The longest `http(s)` image reference accepted, in bytes.
pub const MAX_REMOTE_IMAGE_URL_BYTES: usize = 2048;

/// The most redirects a fetcher follows for one image.
pub const MAX_REMOTE_IMAGE_REDIRECTS: usize = 3;

/// The default per-image deadline of a remote fetch, covering name
/// resolution, connect, TLS, the response head and the whole body.
pub const DEFAULT_REMOTE_IMAGE_TIMEOUT_MS: u64 = 10_000;

/// How many characters of a reference an error message echoes.
pub const SHOWN_URL_CHARS: usize = 96;

/// `url` truncated to [`SHOWN_URL_CHARS`] characters, as every remote-image
/// error message echoes it — with any credentials (`user:password@`) in its
/// authority masked as `***@`, so a message (or a log that records it) never
/// repeats a secret.
#[must_use]
pub fn shown_url(url: &str) -> String {
    mask_userinfo(url).chars().take(SHOWN_URL_CHARS).collect()
}

/// `url` with the userinfo of its authority replaced by `***`.
fn mask_userinfo(url: &str) -> std::borrow::Cow<'_, str> {
    let Some(scheme_end) = url.find("://") else {
        return std::borrow::Cow::Borrowed(url);
    };
    let rest = &url[scheme_end + 3..];
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    match rest[..authority_end].rfind('@') {
        Some(at) => std::borrow::Cow::Owned(format!("{}://***{}", &url[..scheme_end], &rest[at..])),
        None => std::borrow::Cow::Borrowed(url),
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  The fetcher seam
// ─────────────────────────────────────────────────────────────────────────

/// Fetches the encoded bytes of a remote image, synchronously.
///
/// An implementation applies the policy of this module (or a stricter one)
/// to every hop it makes and returns the bytes of a `200` response, or the
/// typed refusal / failure: [`ImageInputError::RemoteUrlRefused`] for a URL
/// the policy refuses, [`ImageInputError::RemoteFetchFailed`] for a transfer
/// that failed, [`ImageInputError::EncodedTooLarge`] for a body over the cap.
/// It is called from whatever thread resolves the reference — a command's
/// main thread, a blocking-pool thread of a server — so it must not assume
/// an async runtime and must bound its own wait.
pub trait RemoteImageFetcher: Send + Sync {
    /// Fetch the image `url` names (a trimmed `http` / `https` reference,
    /// exactly as the request gave it), reading at most `max_bytes` bytes of
    /// body.
    ///
    /// # Errors
    ///
    /// [`ImageInputError::RemoteUrlRefused`],
    /// [`ImageInputError::RemoteFetchFailed`] or
    /// [`ImageInputError::EncodedTooLarge`].
    fn fetch(&self, url: &str, max_bytes: usize) -> Result<Vec<u8>, ImageInputError>;

    /// Check `url` without any network activity: everything the fetcher
    /// can decide before a name is resolved. The default checks the URL
    /// syntax ([`parse_remote_image_url`]); a fetcher with an allowlist adds
    /// the literal-address and `localhost` rules ([`vet_target`]).
    ///
    /// # Errors
    ///
    /// [`ImageInputError::RemoteUrlRefused`].
    fn preflight(&self, url: &str) -> Result<(), ImageInputError> {
        parse_remote_image_url(url)
            .map(drop)
            .map_err(|refusal| refusal.into_error(url))
    }

    /// One line saying what the fetcher fetches (deadline, allowlist), for
    /// logs and `Debug` output.
    fn describe(&self) -> String {
        "a remote image fetcher installed by the application".to_string()
    }
}

/// A shared [`RemoteImageFetcher`], compared by identity: two handles are
/// equal when they point at the same fetcher. This is what lets
/// [`super::ImageSourcePolicy`] stay `Clone + PartialEq + Eq + Debug`.
#[derive(Clone)]
pub struct SharedRemoteImageFetcher(Arc<dyn RemoteImageFetcher>);

impl SharedRemoteImageFetcher {
    /// Share `fetcher`.
    #[must_use]
    pub fn new(fetcher: Arc<dyn RemoteImageFetcher>) -> Self {
        Self(fetcher)
    }

    /// The fetcher.
    #[must_use]
    pub fn fetcher(&self) -> &dyn RemoteImageFetcher {
        self.0.as_ref()
    }

    /// The fetcher as the shared pointer it is held by.
    #[must_use]
    pub fn as_arc(&self) -> &Arc<dyn RemoteImageFetcher> {
        &self.0
    }
}

impl<F: RemoteImageFetcher + 'static> From<Arc<F>> for SharedRemoteImageFetcher {
    fn from(fetcher: Arc<F>) -> Self {
        Self(fetcher)
    }
}

impl fmt::Debug for SharedRemoteImageFetcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SharedRemoteImageFetcher")
            .field(&self.0.describe())
            .finish()
    }
}

impl PartialEq for SharedRemoteImageFetcher {
    fn eq(&self, other: &Self) -> bool {
        // `Arc::ptr_eq` compares the data addresses and ignores the vtable.
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SharedRemoteImageFetcher {}

/// How `http(s)` image references are treated (see the module docs, "Three
/// states").
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RemoteImageAccess {
    /// No opt-in (the default): refused with `image_url_fetch_disabled`
    /// before anything is opened or resolved.
    #[default]
    Disabled,
    /// The operator opted in, but the application installed no fetcher:
    /// refused with `image_url_fetch_disabled`, saying so.
    OptedInWithoutFetcher,
    /// Fetched through this fetcher.
    Fetcher(SharedRemoteImageFetcher),
}

impl RemoteImageAccess {
    /// Whether the operator opted in (with or without a fetcher installed).
    #[must_use]
    pub fn is_opted_in(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// The installed fetcher, if any.
    #[must_use]
    pub fn fetcher(&self) -> Option<&SharedRemoteImageFetcher> {
        match self {
            Self::Fetcher(fetcher) => Some(fetcher),
            Self::Disabled | Self::OptedInWithoutFetcher => None,
        }
    }
}

/// Why a reference with no opt-in is refused.
const NOT_OPTED_IN_REASON: &str =
    "fetching them would let a request make this process open network connections of its \
     choosing (server-side request forgery), so remote image URLs are disabled unless the \
     operator opts in (--allow-image-url-fetch or OXI_ALLOW_IMAGE_URL_FETCH=1) on a front end \
     that installs a fetcher with an address policy, such as the `oxibonsai` command";

/// Why a reference is refused when the opt-in was given but nothing can
/// fetch it.
const OPTED_IN_WITHOUT_FETCHER_REASON: &str =
    "remote fetching was opted in to, but the application resolving this request installed \
     no fetcher with an address policy (ImageSourcePolicy::with_remote_fetcher), so nothing \
     can fetch it";

/// Resolve an `http(s)` reference under `access`: the typed refusal, or the
/// installed fetcher's answer, held to `max_bytes` even if the fetcher is not.
pub(crate) fn fetch_remote_reference(
    reference: &str,
    access: &RemoteImageAccess,
    max_bytes: usize,
) -> Result<Vec<u8>, ImageInputError> {
    let refused = |reason: &str| ImageInputError::RemoteFetchRefused {
        url: shown_url(reference),
        reason: reason.to_string(),
    };
    match access {
        RemoteImageAccess::Disabled => Err(refused(NOT_OPTED_IN_REASON)),
        RemoteImageAccess::OptedInWithoutFetcher => Err(refused(OPTED_IN_WITHOUT_FETCHER_REASON)),
        RemoteImageAccess::Fetcher(fetcher) => {
            let bytes = fetcher.fetcher().fetch(reference, max_bytes)?;
            if bytes.len() > max_bytes {
                return Err(ImageInputError::EncodedTooLarge {
                    bytes: bytes.len(),
                    limit: max_bytes,
                });
            }
            Ok(bytes)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Typed refusals and failures
// ─────────────────────────────────────────────────────────────────────────

/// Why the policy refuses a remote image URL (an `image_url_refused`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RemoteUrlRefusal {
    /// Longer than [`MAX_REMOTE_IMAGE_URL_BYTES`].
    TooLong {
        /// The URL's length in bytes.
        bytes: usize,
    },
    /// Not an absolute `scheme://host...` URL, or it contains a character
    /// no URL may carry here (whitespace, a control character, a backslash).
    Malformed {
        /// What is wrong.
        reason: String,
    },
    /// A scheme other than `http` / `https`.
    Scheme {
        /// The scheme as given (at most 16 characters).
        scheme: String,
    },
    /// Credentials (`user:password@`) in the authority.
    Userinfo,
    /// No host.
    EmptyHost,
    /// A host that is neither a valid DNS name nor a valid IP literal.
    InvalidHost {
        /// What is wrong.
        reason: String,
    },
    /// An IPv6 literal with a zone identifier (`[fe80::1%25en0]`).
    ZoneId,
    /// A port outside `1..=65535` or not a number.
    InvalidPort,
    /// `localhost` or a `*.localhost` name that is not allowlisted.
    LocalhostName {
        /// The host.
        host: String,
    },
    /// The host is (or resolves to) an address outside the public unicast
    /// space and is not allowlisted.
    NotPublic {
        /// The host as the URL names it (never a resolved address).
        host: String,
        /// The class of a literal address; `None` for a resolved name, whose
        /// addresses are never disclosed.
        class: Option<AddressClass>,
    },
    /// A redirect from `https` to `http`.
    Downgrade,
    /// The resolver was asked for a name other than the vetted host.
    UnexpectedHost,
}

impl RemoteUrlRefusal {
    /// This refusal as the [`ImageInputError::RemoteUrlRefused`] of the
    /// reference `url`.
    #[must_use]
    pub fn into_error(self, url: &str) -> ImageInputError {
        ImageInputError::RemoteUrlRefused {
            url: shown_url(url),
            reason: self.to_string(),
        }
    }
}

/// The allowlist flag and variable, for refusal texts.
const ALLOWLIST_HINT: &str =
    "it is not allowlisted (--image-url-allow-host or OXI_IMAGE_URL_ALLOW_HOSTS)";

impl fmt::Display for RemoteUrlRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLong { bytes } => write!(
                f,
                "the URL is {bytes} bytes long; at most {MAX_REMOTE_IMAGE_URL_BYTES} are accepted"
            ),
            Self::Malformed { reason } => write!(f, "malformed URL: {reason}"),
            Self::Scheme { scheme } => write!(
                f,
                "the '{scheme}' scheme is not fetched; only http and https are"
            ),
            Self::Userinfo => {
                f.write_str("a URL carrying credentials (user:password@) is not fetched")
            }
            Self::EmptyHost => f.write_str("the URL names no host"),
            Self::InvalidHost { reason } => write!(f, "invalid host: {reason}"),
            Self::ZoneId => f.write_str("an IPv6 zone identifier (%...) is not accepted"),
            Self::InvalidPort => f.write_str("invalid port: a port is a number in 1-65535"),
            Self::LocalhostName { host } => write!(
                f,
                "the host {host} names this machine (localhost) and {ALLOWLIST_HINT}"
            ),
            Self::NotPublic {
                host,
                class: Some(class),
            } => write!(
                f,
                "the host {host} is not a public address ({class}) and {ALLOWLIST_HINT}"
            ),
            Self::NotPublic { host, class: None } => write!(
                f,
                "the host {host} resolves to an address that is not public, and {ALLOWLIST_HINT}"
            ),
            Self::Downgrade => f.write_str("a redirect from https to http is not followed"),
            Self::UnexpectedHost => {
                f.write_str("the connection asked to resolve a host other than the one vetted")
            }
        }
    }
}

/// Which step of a permitted fetch failed (an `image_url_fetch_failed`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RemoteFetchFailure {
    /// The host name did not resolve, or resolved to no address.
    Resolve,
    /// No connection could be made (an `http` URL).
    Connect,
    /// The connection or its TLS handshake failed (an `https` URL; the
    /// HTTP client reports the two as one connect error).
    ConnectOrTls,
    /// The connection was made but no valid HTTP response came back (the
    /// server closed it, or answered with something that is not HTTP).
    Response,
    /// The server answered (after redirects) with this status; `200` is
    /// the only success.
    Status(u16),
    /// A redirect could not be followed.
    Redirect {
        /// Why.
        reason: String,
    },
    /// The response body failed or ended early.
    Body,
    /// The per-image deadline expired.
    TimedOut {
        /// The deadline.
        ms: u64,
    },
    /// The fetcher could not run at all.
    Unavailable {
        /// Why.
        reason: String,
    },
    /// The request the image belongs to was abandoned (its client went
    /// away, or its deadline expired) before this image was fetched.
    Abandoned,
}

impl RemoteFetchFailure {
    /// This failure as the [`ImageInputError::RemoteFetchFailed`] of the
    /// reference `url`.
    #[must_use]
    pub fn into_error(self, url: &str) -> ImageInputError {
        ImageInputError::RemoteFetchFailed {
            url: shown_url(url),
            reason: self.to_string(),
        }
    }
}

impl fmt::Display for RemoteFetchFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolve => f.write_str("resolve: the host name did not resolve to an address"),
            Self::Connect => f.write_str("connect: no connection could be made"),
            Self::ConnectOrTls => {
                f.write_str("connect/TLS: the connection or its TLS handshake failed")
            }
            Self::Response => f.write_str("response: the server sent no valid HTTP response"),
            Self::Status(status) => write!(f, "status {status}: the server did not answer 200 OK"),
            Self::Redirect { reason } => write!(f, "redirect: {reason}"),
            Self::Body => f.write_str("body: the response body failed or ended early"),
            Self::TimedOut { ms } => write!(f, "timed out after {ms} ms"),
            Self::Unavailable { reason } => write!(f, "the image fetcher cannot run: {reason}"),
            Self::Abandoned => {
                f.write_str("the request was abandoned before this image was fetched")
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  The address policy
// ─────────────────────────────────────────────────────────────────────────

/// A class of addresses no remote image is fetched from (see the module
/// docs for the ranges).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AddressClass {
    /// `0.0.0.0/8` ("this network") or `::`.
    Unspecified,
    /// `127/8` or `::1`.
    Loopback,
    /// `10/8`, `172.16/12`, `192.168/16`.
    Private,
    /// `100.64/10` (carrier-grade NAT).
    SharedAddressSpace,
    /// `169.254/16` or `fe80::/10`.
    LinkLocal,
    /// `192.0.0.0/24` or `2001::/23`.
    ProtocolAssignments,
    /// `192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`, `2001:db8::/32`
    /// or `3fff::/20`.
    Documentation,
    /// `192.88.99.0/24`.
    SixToFourRelay,
    /// `198.18/15` or `2001:2::/48`.
    Benchmarking,
    /// `224/4` or `ff00::/8`.
    Multicast,
    /// `255.255.255.255`.
    Broadcast,
    /// `240/4`, or IPv6 outside `2000::/3`.
    Reserved,
    /// `::ffff:0:0/96`, whatever IPv4 address it embeds.
    Ipv4Mapped,
    /// `::/96` (deprecated IPv4-compatible), whatever IPv4 address it embeds.
    Ipv4Compatible,
    /// `64:ff9b::/96` or `64:ff9b:1::/48`.
    Nat64,
    /// `100::/64`.
    DiscardOnly,
    /// `2001::/32`.
    Teredo,
    /// `2002::/16`.
    SixToFour,
    /// `fc00::/7`.
    UniqueLocal,
    /// `fec0::/10`.
    SiteLocal,
}

impl fmt::Display for AddressClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Unspecified => "unspecified",
            Self::Loopback => "loopback",
            Self::Private => "private",
            Self::SharedAddressSpace => "shared address space",
            Self::LinkLocal => "link-local",
            Self::ProtocolAssignments => "protocol assignments",
            Self::Documentation => "documentation",
            Self::SixToFourRelay => "6to4 relay anycast",
            Self::Benchmarking => "benchmarking",
            Self::Multicast => "multicast",
            Self::Broadcast => "broadcast",
            Self::Reserved => "reserved",
            Self::Ipv4Mapped => "IPv4-mapped",
            Self::Ipv4Compatible => "IPv4-compatible",
            Self::Nat64 => "NAT64",
            Self::DiscardOnly => "discard-only",
            Self::Teredo => "Teredo",
            Self::SixToFour => "6to4",
            Self::UniqueLocal => "unique local",
            Self::SiteLocal => "site-local",
        })
    }
}

/// `true` when the top `bits` bits of `address` equal those of `network`.
fn v4_in(address: u32, network: [u8; 4], bits: u32) -> bool {
    let mask = u32::MAX.checked_shl(32 - bits).unwrap_or(0);
    (address & mask) == (u32::from_be_bytes(network) & mask)
}

/// `true` when the top `bits` bits of `address` equal those of `network`.
fn v6_in(address: u128, network: [u16; 8], bits: u32) -> bool {
    let mask = u128::MAX.checked_shl(128 - bits).unwrap_or(0);
    (address & mask) == (u128::from(Ipv6Addr::from(network)) & mask)
}

/// The class an IPv4 address is refused for, or `None` when it is public.
fn classify_v4(address: Ipv4Addr) -> Option<AddressClass> {
    let a = u32::from(address);
    let table: [([u8; 4], u32, AddressClass); 15] = [
        ([0, 0, 0, 0], 8, AddressClass::Unspecified),
        ([10, 0, 0, 0], 8, AddressClass::Private),
        ([100, 64, 0, 0], 10, AddressClass::SharedAddressSpace),
        ([127, 0, 0, 0], 8, AddressClass::Loopback),
        ([169, 254, 0, 0], 16, AddressClass::LinkLocal),
        ([172, 16, 0, 0], 12, AddressClass::Private),
        ([192, 0, 0, 0], 24, AddressClass::ProtocolAssignments),
        ([192, 0, 2, 0], 24, AddressClass::Documentation),
        ([192, 88, 99, 0], 24, AddressClass::SixToFourRelay),
        ([192, 168, 0, 0], 16, AddressClass::Private),
        ([198, 18, 0, 0], 15, AddressClass::Benchmarking),
        ([198, 51, 100, 0], 24, AddressClass::Documentation),
        ([203, 0, 113, 0], 24, AddressClass::Documentation),
        ([224, 0, 0, 0], 4, AddressClass::Multicast),
        ([255, 255, 255, 255], 32, AddressClass::Broadcast),
    ];
    if let Some((_, _, class)) = table
        .iter()
        .find(|(network, bits, _)| v4_in(a, *network, *bits))
    {
        return Some(*class);
    }
    v4_in(a, [240, 0, 0, 0], 4).then_some(AddressClass::Reserved)
}

/// The class an IPv6 address is refused for, or `None` when it is public.
fn classify_v6(address: Ipv6Addr) -> Option<AddressClass> {
    let a = u128::from(address);
    if a == 0 {
        return Some(AddressClass::Unspecified);
    }
    if a == 1 {
        return Some(AddressClass::Loopback);
    }
    // Ordered most specific first: the named sub-ranges of 2001::/23 before
    // the block itself.
    let table: [([u16; 8], u32, AddressClass); 16] = [
        ([0, 0, 0, 0, 0, 0xffff, 0, 0], 96, AddressClass::Ipv4Mapped),
        ([0; 8], 96, AddressClass::Ipv4Compatible),
        ([0x64, 0xff9b, 0, 0, 0, 0, 0, 0], 96, AddressClass::Nat64),
        ([0x64, 0xff9b, 1, 0, 0, 0, 0, 0], 48, AddressClass::Nat64),
        ([0x100, 0, 0, 0, 0, 0, 0, 0], 64, AddressClass::DiscardOnly),
        ([0x2001, 0, 0, 0, 0, 0, 0, 0], 32, AddressClass::Teredo),
        (
            [0x2001, 2, 0, 0, 0, 0, 0, 0],
            48,
            AddressClass::Benchmarking,
        ),
        (
            [0x2001, 0xdb8, 0, 0, 0, 0, 0, 0],
            32,
            AddressClass::Documentation,
        ),
        (
            [0x2001, 0, 0, 0, 0, 0, 0, 0],
            23,
            AddressClass::ProtocolAssignments,
        ),
        ([0x2002, 0, 0, 0, 0, 0, 0, 0], 16, AddressClass::SixToFour),
        (
            [0x3fff, 0, 0, 0, 0, 0, 0, 0],
            20,
            AddressClass::Documentation,
        ),
        ([0xfc00, 0, 0, 0, 0, 0, 0, 0], 7, AddressClass::UniqueLocal),
        ([0xfe80, 0, 0, 0, 0, 0, 0, 0], 10, AddressClass::LinkLocal),
        ([0xfec0, 0, 0, 0, 0, 0, 0, 0], 10, AddressClass::SiteLocal),
        ([0xff00, 0, 0, 0, 0, 0, 0, 0], 8, AddressClass::Multicast),
        ([0x2000, 0, 0, 0, 0, 0, 0, 0], 3, AddressClass::Reserved),
    ];
    for (network, bits, class) in table {
        if v6_in(a, network, bits) {
            // `2000::/3` is the global-unicast block: inside it is public
            // (its refused sub-ranges matched above), outside it is not.
            return (class != AddressClass::Reserved).then_some(class);
        }
    }
    Some(AddressClass::Reserved)
}

/// The class `address` is refused for, or `None` when it is a public
/// unicast address a remote image may be fetched from (see the module docs
/// for the table).
#[must_use]
pub fn classify_address(address: IpAddr) -> Option<AddressClass> {
    match address {
        IpAddr::V4(v4) => classify_v4(v4),
        IpAddr::V6(v6) => classify_v6(v6),
    }
}

/// Whether `address` is a public unicast address.
#[must_use]
pub fn is_public_address(address: IpAddr) -> bool {
    classify_address(address).is_none()
}

/// Whether `name` (a lower-case DNS name without a trailing dot) is
/// `localhost` or a name under it.
#[must_use]
pub fn is_localhost_name(name: &str) -> bool {
    name == "localhost" || name.ends_with(".localhost")
}

// ─────────────────────────────────────────────────────────────────────────
//  URL syntax
// ─────────────────────────────────────────────────────────────────────────

/// The two schemes a remote image is fetched over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemoteScheme {
    /// `http`.
    Http,
    /// `https`.
    Https,
}

impl RemoteScheme {
    /// `"http"` or `"https"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Https => "https",
        }
    }

    /// `80` or `443`.
    #[must_use]
    pub const fn default_port(self) -> u16 {
        match self {
            Self::Http => 80,
            Self::Https => 443,
        }
    }
}

/// A URL's host, canonical: an IP literal by its address, a DNS name
/// lower-cased without its trailing dot.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RemoteHost {
    /// A DNS name.
    Domain(String),
    /// An IPv4 literal (in any of the spellings the module docs list).
    Ipv4(Ipv4Addr),
    /// A bracketed IPv6 literal.
    Ipv6(Ipv6Addr),
}

impl RemoteHost {
    /// The address of an IP literal.
    #[must_use]
    pub fn ip(&self) -> Option<IpAddr> {
        match self {
            Self::Domain(_) => None,
            Self::Ipv4(v4) => Some(IpAddr::V4(*v4)),
            Self::Ipv6(v6) => Some(IpAddr::V6(*v6)),
        }
    }

    /// The name a resolver is asked for: the DNS name, or the literal
    /// address without brackets.
    #[must_use]
    pub fn resolver_name(&self) -> String {
        match self {
            Self::Domain(name) => name.clone(),
            Self::Ipv4(v4) => v4.to_string(),
            Self::Ipv6(v6) => v6.to_string(),
        }
    }
}

impl fmt::Display for RemoteHost {
    /// As it appears in a URL's authority (an IPv6 literal bracketed).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Domain(name) => f.write_str(name),
            Self::Ipv4(v4) => write!(f, "{v4}"),
            Self::Ipv6(v6) => write!(f, "[{v6}]"),
        }
    }
}

/// A remote image URL that passed the syntax rules, in canonical parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteImageUrl {
    scheme: RemoteScheme,
    host: RemoteHost,
    port: u16,
    explicit_port: bool,
    /// Starts with `/`; every byte a request target may carry.
    path: String,
    /// Without the `?`.
    query: Option<String>,
}

impl RemoteImageUrl {
    /// `http` or `https`.
    #[must_use]
    pub fn scheme(&self) -> RemoteScheme {
        self.scheme
    }

    /// The host.
    #[must_use]
    pub fn host(&self) -> &RemoteHost {
        &self.host
    }

    /// The port a connection goes to (the scheme's default unless the URL
    /// names one).
    #[must_use]
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The path, starting with `/`.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The query, without its `?`.
    #[must_use]
    pub fn query(&self) -> Option<&str> {
        self.query.as_deref()
    }

    /// The authority a request names: the canonical host, and the port when
    /// the URL named one.
    #[must_use]
    pub fn authority(&self) -> String {
        if self.explicit_port {
            format!("{}:{}", self.host, self.port)
        } else {
            self.host.to_string()
        }
    }

    /// `scheme://host:port` — what a log line may record of the URL (never
    /// its path or query, which can carry credentials).
    #[must_use]
    pub fn origin(&self) -> String {
        format!("{}://{}:{}", self.scheme.as_str(), self.host, self.port)
    }

    /// The absolute URL a request is sent to, rebuilt from the vetted parts
    /// (no fragment, no credentials, the canonical host).
    #[must_use]
    pub fn request_target(&self) -> String {
        let mut target = format!(
            "{}://{}{}",
            self.scheme.as_str(),
            self.authority(),
            self.path
        );
        if let Some(query) = &self.query {
            target.push('?');
            target.push_str(query);
        }
        target
    }

    /// Resolve a redirect's `Location` against this URL (RFC 3986 §5.2:
    /// an absolute URL, a scheme-relative `//host/...`, an absolute path, a
    /// query, or a relative path merged with this URL's directory, dot
    /// segments removed) and parse the result under every rule of
    /// [`parse_remote_image_url`] again.
    ///
    /// # Errors
    ///
    /// The [`RemoteUrlRefusal`] of the resolved URL, or
    /// [`RemoteUrlRefusal::Malformed`] for an empty `Location`.
    pub fn join(&self, location: &str) -> Result<RemoteImageUrl, RemoteUrlRefusal> {
        let location = location.trim();
        if location.is_empty() {
            return Err(RemoteUrlRefusal::Malformed {
                reason: "an empty redirect location".to_string(),
            });
        }
        check_characters(location)?;
        if has_scheme(location) {
            return parse_remote_image_url(location);
        }
        if location.starts_with("//") {
            return parse_remote_image_url(&format!("{}:{location}", self.scheme.as_str()));
        }
        let without_fragment = location.split('#').next().unwrap_or_default();
        let (path_part, query) = match without_fragment.split_once('?') {
            Some((path, query)) => (path, Some(query)),
            None => (without_fragment, None),
        };
        let (path, query) = if path_part.is_empty() {
            // `?query` keeps this path; a fragment-only reference is this URL.
            (
                self.path.clone(),
                if without_fragment.is_empty() {
                    self.query.clone()
                } else {
                    query.map(str::to_string)
                },
            )
        } else if path_part.starts_with('/') {
            (remove_dot_segments(path_part), query.map(str::to_string))
        } else {
            let directory = match self.path.rfind('/') {
                Some(slash) => &self.path[..=slash],
                None => "/",
            };
            (
                remove_dot_segments(&format!("{directory}{path_part}")),
                query.map(str::to_string),
            )
        };
        let mut target = format!("{}://{}{path}", self.scheme.as_str(), self.authority());
        if let Some(query) = query {
            target.push('?');
            target.push_str(&query);
        }
        parse_remote_image_url(&target)
    }
}

/// `true` for a reference that starts with a URI scheme (`alpha *( alpha /
/// digit / "+" / "-" / "." ) ":"`, the colon before any `/`, `?` or `#`).
fn has_scheme(reference: &str) -> bool {
    let end = reference.find(['/', '?', '#']).unwrap_or(reference.len());
    match reference[..end].split_once(':') {
        Some((scheme, _)) => {
            let mut chars = scheme.chars();
            chars.next().is_some_and(|c| c.is_ascii_alphabetic())
                && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        }
        None => false,
    }
}

/// RFC 3986 §5.2.4 on an absolute path.
fn remove_dot_segments(path: &str) -> String {
    let segments: Vec<&str> = path.split('/').skip(1).collect();
    let last = segments.len().saturating_sub(1);
    let mut out: Vec<&str> = Vec::with_capacity(segments.len());
    let mut trailing_slash = false;
    for (index, segment) in segments.iter().enumerate() {
        match *segment {
            "." => trailing_slash |= index == last,
            ".." => {
                out.pop();
                trailing_slash |= index == last;
            }
            other => out.push(other),
        }
    }
    let mut result = format!("/{}", out.join("/"));
    if trailing_slash && !result.ends_with('/') {
        result.push('/');
    }
    result
}

/// Refuse whitespace, control characters and backslashes anywhere.
fn check_characters(text: &str) -> Result<(), RemoteUrlRefusal> {
    if let Some(bad) = text
        .chars()
        .find(|c| c.is_whitespace() || c.is_control() || *c == '\\')
    {
        let what = if bad == '\\' {
            "a backslash".to_string()
        } else {
            format!("the character U+{:04X}", u32::from(bad))
        };
        return Err(RemoteUrlRefusal::Malformed {
            reason: format!("{what} is not accepted in an image URL"),
        });
    }
    Ok(())
}

/// Percent-encode every byte a request target may not carry as-is (RFC 3986
/// `pchar`, `/`, and in a query `?`; an existing `%XX` is kept).
fn encode_target(text: &str, query: bool) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        let keep = byte.is_ascii_alphanumeric()
            || matches!(
                byte,
                b'-' | b'.'
                    | b'_'
                    | b'~'
                    | b'!'
                    | b'$'
                    | b'&'
                    | b'\''
                    | b'('
                    | b')'
                    | b'*'
                    | b'+'
                    | b','
                    | b';'
                    | b'='
                    | b':'
                    | b'@'
                    | b'/'
                    | b'%'
            )
            || (query && byte == b'?');
        if keep {
            out.push(char::from(byte));
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// Parse one number of an IPv4 literal the way `inet_aton` (and the WHATWG
/// URL standard) do: `0x`/`0X` hexadecimal (digits may be empty), a
/// leading `0` octal, decimal otherwise.
fn parse_ipv4_number(part: &str) -> Option<u64> {
    if part.is_empty() {
        return None;
    }
    let (digits, radix) =
        if let Some(hex) = part.strip_prefix("0x").or_else(|| part.strip_prefix("0X")) {
            (hex, 16)
        } else if part.len() > 1 && part.starts_with('0') {
            (&part[1..], 8)
        } else {
            (part, 10)
        };
    if digits.is_empty() {
        return Some(0);
    }
    if digits.len() > 16 || !digits.chars().all(|c| c.is_digit(radix)) {
        return None;
    }
    u64::from_str_radix(digits, radix).ok()
}

/// `true` when a host "ends in a number" (WHATWG): its last label (after one
/// trailing dot) is all decimal digits, or `0x` followed by hexadecimal
/// digits — such a host is an IPv4 literal or invalid, never a name.
fn ends_in_number(host: &str) -> bool {
    let host = host.strip_suffix('.').unwrap_or(host);
    let last = host.rsplit('.').next().unwrap_or_default();
    if !last.is_empty() && last.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    last.strip_prefix("0x")
        .or_else(|| last.strip_prefix("0X"))
        .is_some_and(|hex| hex.chars().all(|c| c.is_ascii_hexdigit()))
}

/// The WHATWG IPv4 parser (one to four numbers; the last fills the
/// remaining bytes).
fn parse_ipv4_literal(host: &str) -> Option<Ipv4Addr> {
    let host = host.strip_suffix('.').unwrap_or(host);
    let parts: Vec<&str> = host.split('.').collect();
    if parts.is_empty() || parts.len() > 4 {
        return None;
    }
    let numbers: Vec<u64> = parts
        .iter()
        .map(|part| parse_ipv4_number(part))
        .collect::<Option<Vec<u64>>>()?;
    let (last, leading) = numbers.split_last()?;
    if leading.iter().any(|n| *n > 255) {
        return None;
    }
    let remaining_bytes = 5 - numbers.len();
    let limit = 1u64 << (8 * remaining_bytes as u32);
    if *last >= limit {
        return None;
    }
    let mut value = *last;
    for (index, n) in leading.iter().enumerate() {
        value += n << (8 * (3 - index as u32));
    }
    u32::try_from(value).ok().map(Ipv4Addr::from)
}

/// A DNS name's syntax: labels of 1-63 letters, digits, `-` or `_`, at most
/// 253 characters in all.
fn check_domain(name: &str) -> Result<(), RemoteUrlRefusal> {
    let invalid = |reason: &str| RemoteUrlRefusal::InvalidHost {
        reason: reason.to_string(),
    };
    if !name.is_ascii() {
        return Err(invalid(
            "an internationalised host name must be given in its ASCII (xn--) form",
        ));
    }
    if name.len() > 253 {
        return Err(invalid("a host name is at most 253 characters"));
    }
    for label in name.split('.') {
        if label.is_empty() || label.len() > 63 {
            return Err(invalid("a host name label is 1 to 63 characters"));
        }
        if !label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(invalid(
                "a host name holds only letters, digits, '-', '_' and '.'",
            ));
        }
    }
    Ok(())
}

/// Parse a host as a URL's authority (or an allowlist entry) spells it:
/// a bracketed IPv6 literal, an IPv4 literal in any `inet_aton` spelling, or
/// a DNS name.
fn parse_host(text: &str) -> Result<RemoteHost, RemoteUrlRefusal> {
    if text.is_empty() {
        return Err(RemoteUrlRefusal::EmptyHost);
    }
    if let Some(inner) = text.strip_prefix('[') {
        let inner = inner
            .strip_suffix(']')
            .ok_or_else(|| RemoteUrlRefusal::InvalidHost {
                reason: "an unterminated IPv6 literal".to_string(),
            })?;
        if inner.contains('%') {
            return Err(RemoteUrlRefusal::ZoneId);
        }
        return inner
            .parse::<Ipv6Addr>()
            .map(RemoteHost::Ipv6)
            .map_err(|_| RemoteUrlRefusal::InvalidHost {
                reason: "not a valid IPv6 address".to_string(),
            });
    }
    if text.contains(['[', ']', '%']) {
        return Err(RemoteUrlRefusal::InvalidHost {
            reason: "brackets and percent-encoding are not accepted in a host name".to_string(),
        });
    }
    let lower = text.to_ascii_lowercase();
    if ends_in_number(&lower) {
        return parse_ipv4_literal(&lower)
            .map(RemoteHost::Ipv4)
            .ok_or_else(|| RemoteUrlRefusal::InvalidHost {
                reason: "not a valid IPv4 address".to_string(),
            });
    }
    let name = lower.strip_suffix('.').unwrap_or(&lower);
    if name.is_empty() {
        return Err(RemoteUrlRefusal::EmptyHost);
    }
    check_domain(name)?;
    Ok(RemoteHost::Domain(name.to_string()))
}

/// Parse a port: digits only, `1..=65535`.
fn parse_port(text: &str) -> Result<u16, RemoteUrlRefusal> {
    if text.is_empty() || text.len() > 5 || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(RemoteUrlRefusal::InvalidPort);
    }
    match text.parse::<u16>() {
        Ok(port) if port > 0 => Ok(port),
        _ => Err(RemoteUrlRefusal::InvalidPort),
    }
}

/// Split an authority (no userinfo) into its host and optional port.
fn split_host_port(authority: &str) -> Result<(RemoteHost, Option<u16>), RemoteUrlRefusal> {
    if authority.starts_with('[') {
        let close = authority
            .find(']')
            .ok_or_else(|| RemoteUrlRefusal::InvalidHost {
                reason: "an unterminated IPv6 literal".to_string(),
            })?;
        let host = parse_host(&authority[..=close])?;
        let rest = &authority[close + 1..];
        let port = match rest.strip_prefix(':') {
            Some(port) => Some(parse_port(port)?),
            None if rest.is_empty() => None,
            None => {
                return Err(RemoteUrlRefusal::InvalidHost {
                    reason: "unexpected text after an IPv6 literal".to_string(),
                })
            }
        };
        return Ok((host, port));
    }
    match authority.split_once(':') {
        Some((_, port)) if port.contains(':') => Err(RemoteUrlRefusal::InvalidHost {
            reason: "an IPv6 literal must be bracketed".to_string(),
        }),
        Some((host, port)) => Ok((parse_host(host)?, Some(parse_port(port)?))),
        None => Ok((parse_host(authority)?, None)),
    }
}

/// Parse an `http(s)` image reference under the syntax rules of the module
/// docs.
///
/// # Errors
///
/// The [`RemoteUrlRefusal`] naming the rule the URL breaks.
pub fn parse_remote_image_url(url: &str) -> Result<RemoteImageUrl, RemoteUrlRefusal> {
    let url = url.trim();
    if url.len() > MAX_REMOTE_IMAGE_URL_BYTES {
        return Err(RemoteUrlRefusal::TooLong { bytes: url.len() });
    }
    check_characters(url)?;
    let (scheme_text, rest) = url
        .split_once("://")
        .ok_or_else(|| RemoteUrlRefusal::Malformed {
            reason: "not an absolute http:// or https:// URL".to_string(),
        })?;
    let scheme = if scheme_text.eq_ignore_ascii_case("http") {
        RemoteScheme::Http
    } else if scheme_text.eq_ignore_ascii_case("https") {
        RemoteScheme::Https
    } else {
        return Err(RemoteUrlRefusal::Scheme {
            scheme: scheme_text.chars().take(16).collect(),
        });
    };
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(authority_end);
    if authority.contains('@') {
        return Err(RemoteUrlRefusal::Userinfo);
    }
    if authority.is_empty() {
        return Err(RemoteUrlRefusal::EmptyHost);
    }
    let (host, port) = split_host_port(authority)?;
    let without_fragment = tail.split('#').next().unwrap_or_default();
    let (path, query) = match without_fragment.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (without_fragment, None),
    };
    let path = if path.is_empty() {
        "/".to_string()
    } else {
        encode_target(path, false)
    };
    Ok(RemoteImageUrl {
        scheme,
        host,
        port: port.unwrap_or(scheme.default_port()),
        explicit_port: port.is_some(),
        path,
        query: query.map(|q| encode_target(q, true)),
    })
}

// ─────────────────────────────────────────────────────────────────────────
//  The operator's allowlist
// ─────────────────────────────────────────────────────────────────────────

/// One `host[:port]` the operator exempted from the address policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AllowedHost {
    host: RemoteHost,
    port: Option<u16>,
}

impl AllowedHost {
    /// The host.
    #[must_use]
    pub fn host(&self) -> &RemoteHost {
        &self.host
    }

    /// The port, when the entry names one.
    #[must_use]
    pub fn port(&self) -> Option<u16> {
        self.port
    }

    /// Whether `url` is the host (and port, when given) this entry names.
    #[must_use]
    pub fn permits(&self, url: &RemoteImageUrl) -> bool {
        self.host == url.host && self.port.is_none_or(|port| port == url.port)
    }
}

impl fmt::Display for AllowedHost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.port {
            Some(port) => write!(f, "{}:{port}", self.host),
            None => write!(f, "{}", self.host),
        }
    }
}

/// Parse one allowlist entry: `host`, `host:port`, `[v6]`, `[v6]:port`, or
/// an unbracketed IPv6 address (then without a port).
///
/// # Errors
///
/// A message saying what is wrong with the entry (the caller names the
/// entry and the setting it came from).
pub fn parse_allowed_host(entry: &str) -> Result<AllowedHost, String> {
    let entry = entry.trim();
    if entry.is_empty() {
        return Err("an empty entry".to_string());
    }
    if entry.contains("://") {
        return Err("give a host or host:port, not a URL".to_string());
    }
    if entry.contains(['/', '@', '?', '#', '\\']) || entry.chars().any(char::is_whitespace) {
        return Err(
            "a host or host:port holds no '/', '@', '?', '#', backslash or whitespace".to_string(),
        );
    }
    if !entry.starts_with('[') {
        if let Ok(v6) = entry.parse::<Ipv6Addr>() {
            return Ok(AllowedHost {
                host: RemoteHost::Ipv6(v6),
                port: None,
            });
        }
    }
    let (host, port) = split_host_port(entry).map_err(|refusal| refusal.to_string())?;
    Ok(AllowedHost { host, port })
}

/// The operator's allowlist (see the module docs).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostAllowlist {
    entries: Vec<AllowedHost>,
}

impl HostAllowlist {
    /// An allowlist of `entries` (duplicates removed, order kept).
    #[must_use]
    pub fn new(entries: Vec<AllowedHost>) -> Self {
        let mut unique: Vec<AllowedHost> = Vec::with_capacity(entries.len());
        for entry in entries {
            if !unique.contains(&entry) {
                unique.push(entry);
            }
        }
        Self { entries: unique }
    }

    /// Whether the allowlist is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries.
    #[must_use]
    pub fn entries(&self) -> &[AllowedHost] {
        &self.entries
    }

    /// Whether `url`'s host (and port) is allowlisted.
    #[must_use]
    pub fn permits(&self, url: &RemoteImageUrl) -> bool {
        self.entries.iter().any(|entry| entry.permits(url))
    }
}

impl fmt::Display for HostAllowlist {
    /// The entries, comma-separated; `(none)` when empty.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.entries.is_empty() {
            return f.write_str("(none)");
        }
        for (index, entry) in self.entries.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{entry}")?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Vetting
// ─────────────────────────────────────────────────────────────────────────

/// What [`vet_target`] decided about a URL's host before any resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetVerdict {
    /// The operator allowlisted the host (and port): no address class or
    /// name rule applies to it.
    Allowlisted,
    /// A public IP literal: dialled as it is, no resolution.
    PublicLiteral(IpAddr),
    /// A DNS name: every address it resolves to must pass [`vet_resolved`].
    Resolve,
}

/// Decide what can be decided about `url`'s host without resolving it: an
/// allowlisted host passes, an IP literal is classified, `localhost` and
/// `*.localhost` are refused by name, and any other name must be resolved
/// and vetted.
///
/// # Errors
///
/// [`RemoteUrlRefusal::NotPublic`] (with the class) for a refused literal,
/// [`RemoteUrlRefusal::LocalhostName`] for a `localhost` name.
pub fn vet_target(
    url: &RemoteImageUrl,
    allowlist: &HostAllowlist,
) -> Result<TargetVerdict, RemoteUrlRefusal> {
    if allowlist.permits(url) {
        return Ok(TargetVerdict::Allowlisted);
    }
    match &url.host {
        RemoteHost::Domain(name) if is_localhost_name(name) => {
            Err(RemoteUrlRefusal::LocalhostName { host: name.clone() })
        }
        RemoteHost::Domain(_) => Ok(TargetVerdict::Resolve),
        literal => {
            let address = literal.ip().ok_or(RemoteUrlRefusal::UnexpectedHost)?;
            match classify_address(address) {
                Some(class) => Err(RemoteUrlRefusal::NotPublic {
                    host: literal.to_string(),
                    class: Some(class),
                }),
                None => Ok(TargetVerdict::PublicLiteral(address)),
            }
        }
    }
}

/// [`vet_resolved`] with the classification supplied by the caller (a
/// fetcher's tests stand a test classifier in for the public internet).
///
/// # Errors
///
/// As [`vet_resolved`].
pub fn vet_resolved_with(
    host: &RemoteHost,
    addresses: &[IpAddr],
    classify: impl Fn(IpAddr) -> Option<AddressClass>,
) -> Result<(), RemoteUrlRefusal> {
    if addresses.iter().any(|address| classify(*address).is_some()) {
        return Err(RemoteUrlRefusal::NotPublic {
            host: host.to_string(),
            class: None,
        });
    }
    Ok(())
}

/// Vet the addresses a (not allowlisted) host resolved to: every one must be
/// public, or the whole fetch is refused.
///
/// # Errors
///
/// [`RemoteUrlRefusal::NotPublic`] naming the host, never the address.
pub fn vet_resolved(host: &RemoteHost, addresses: &[IpAddr]) -> Result<(), RemoteUrlRefusal> {
    vet_resolved_with(host, addresses, classify_address)
}

#[cfg(test)]
#[path = "remote_tests.rs"]
mod tests;
