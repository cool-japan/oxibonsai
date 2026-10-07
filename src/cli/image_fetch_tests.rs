//! The remote-image fetcher against loopback listeners: the address policy,
//! the redirect rules, the bounds, the allowlist, what is disclosed, and the
//! threads it is called from. Every listener counts the connections it
//! accepts, so a defence that is "no connection" is checked as exactly that.

use super::resolver::LookupFuture;
use super::test_server::{Reply, TestServer};
use super::transfer::{dialled_is_vetted, redirect_target, FetchError};
use super::*;

use std::collections::VecDeque;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use oxibonsai_model::vision::remote::{RemoteUrlRefusal, MAX_REMOTE_IMAGE_URL_BYTES};
use oxibonsai_model::vision::{load_image_bytes, load_image_source, ImageSourcePolicy};

use crate::cli::bonsai2::tests::PATTERN_PNG_DATA_URI;
use crate::cli::bonsai2::{ImageSourceFlags, ALLOW_IMAGE_URL_FETCH_ENV};
use crate::cli::util::test_env::{self, EnvVarGuard};

// ── Fixtures ─────────────────────────────────────────────────────────────

/// The 256 x 192 PNG the vision tests use, as bytes.
fn png() -> Vec<u8> {
    load_image_bytes(PATTERN_PNG_DATA_URI, &ImageSourcePolicy::local_user())
        .expect("the embedded PNG decodes from base64")
}

/// Settings with a deadline of `timeout_ms` and the allowlist `allow`.
fn settings(timeout_ms: u64, allow: &[String]) -> ImageFetchSettings {
    ImageFetchSettings::resolve(Some(timeout_ms), None, allow, None).expect("valid settings")
}

/// Transfer slots of a test's own, sized like the process-wide bound: tests
/// run side by side in one process and must not take each other's slots.
fn own_capacity() -> Arc<capacity::FetchCapacity> {
    capacity::FetchCapacity::new(
        capacity::MAX_FETCHES_IN_FLIGHT,
        capacity::MAX_FETCHES_WAITING,
    )
}

/// A fetcher with the system resolver.
fn fetcher(timeout_ms: u64, allow: &[String]) -> ImageUrlFetcher {
    ImageUrlFetcher::new(settings(timeout_ms, allow)).with_capacity(own_capacity())
}

/// `127.0.0.1:<port>` of a server, as an allowlist entry.
fn allow_v4(server: &TestServer) -> Vec<String> {
    vec![format!("127.0.0.1:{}", server.port())]
}

/// A resolver that answers from a script, one answer per call.
#[derive(Default)]
struct StubLookup {
    answers: Mutex<VecDeque<Vec<IpAddr>>>,
    calls: AtomicUsize,
}

impl StubLookup {
    fn answering(answers: Vec<Vec<IpAddr>>) -> Arc<Self> {
        Arc::new(Self {
            answers: Mutex::new(answers.into()),
            calls: AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl AddressLookup for StubLookup {
    fn lookup(&self, _host: &str) -> LookupFuture {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let answer = self
            .answers
            .lock()
            .ok()
            .and_then(|mut answers| answers.pop_front())
            .unwrap_or_default();
        Box::pin(async move { Ok(answer) })
    }
}

fn stub_fetcher(
    timeout_ms: u64,
    allow: &[String],
    lookup: &Arc<StubLookup>,
    classify: fn(IpAddr) -> Option<AddressClass>,
) -> ImageUrlFetcher {
    let lookup: Arc<dyn AddressLookup> = lookup.clone();
    ImageUrlFetcher::with_parts(settings(timeout_ms, allow), lookup, classify)
        .with_capacity(own_capacity())
}

const V4_LOOPBACK: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
const V6_LOOPBACK: IpAddr = IpAddr::V6(Ipv6Addr::LOCALHOST);
/// A second IPv4 loopback address: where [`same_port_pair`] puts its second
/// listener on a host without an IPv6 loopback.
const V4_SECOND_LOOPBACK: IpAddr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2));
/// Spellings of [`V4_SECOND_LOOPBACK`] that
/// [`every_spelling_of_a_refused_literal_is_refused_before_any_connection`]
/// appends to its list: the `inet_aton` forms the policy reads an IPv4
/// literal in, and the IPv4-mapped forms. Each is refused on every host —
/// the IPv4 ones as loopback (the rule covers all of `127.0.0.0/8`), the
/// IPv4-mapped ones as IPv4-mapped.
const V4_SECOND_LOOPBACK_SPELLINGS: [&str; 12] = [
    "127.0.0.2",
    "127.0.0.2.",
    "2130706434",
    "0x7f.0.0.2",
    "0X7F.0.0.2",
    "0x7f000002",
    "0177.0.0.2",
    "017700000002",
    "127.2",
    "127.0.2",
    "[::ffff:127.0.0.2]",
    "[::FFFF:7F00:2]",
];

/// The test stand-in for the public internet: `127.0.0.1` counts as a
/// public address, everything else is classified by the real policy — so
/// the second loopback address of [`same_port_pair`] (`::1`, or `127.0.0.2`)
/// is refused as loopback here exactly as under [`classify_address`]. Only
/// the DNS-rebinding and mixed-answer tests use it — a real public address
/// cannot be dialled hermetically, and those tests are about which address
/// is dialled, not about the classes.
fn loopback_v4_is_public(address: IpAddr) -> Option<AddressClass> {
    if address == V4_LOOPBACK {
        None
    } else {
        classify_address(address)
    }
}

/// `EAFNOSUPPORT` ("address family not supported") on this target: what a
/// bind fails with when the kernel has no such address family at all (IPv6
/// compiled out, or booted with `ipv6.disable=1`). `std` reports it as an
/// uncategorised [`io::ErrorKind`], so it is matched by its number, which
/// differs per target. The numbers are the `libc` crate's: one for most
/// Linux targets, another for the BSDs and Apple's systems (which SPARC
/// Linux keeps, after SunOS), a third for illumos and Solaris (which MIPS
/// Linux keeps, after IRIX); on Windows, whose sockets report WinSock's
/// codes, it is `WSAEAFNOSUPPORT`. `None` on any other target: a bind
/// failing with it there is not recognised, and panics like any other
/// unexplained bind error ([`bind_outcome`]).
const EAFNOSUPPORT: Option<i32> = if cfg!(target_os = "windows") {
    Some(10_047)
} else if cfg!(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "netbsd",
    target_os = "openbsd",
    all(
        target_os = "linux",
        any(target_arch = "sparc", target_arch = "sparc64")
    )
)) {
    Some(47)
} else if cfg!(any(
    target_os = "illumos",
    target_os = "solaris",
    all(
        target_os = "linux",
        any(
            target_arch = "mips",
            target_arch = "mips64",
            target_arch = "mips32r6",
            target_arch = "mips64r6"
        )
    )
)) {
    Some(124)
} else if cfg!(any(target_os = "linux", target_os = "android")) {
    Some(97)
} else {
    None
};

/// Whether `error`, from binding a listener, is one the helpers below read
/// as "this host lacks the address": `AddrNotAvailable` (no interface has
/// it — `::1` with IPv6 disabled on the loopback interface, `127.0.0.2`
/// where the loopback answers only for `127.0.0.1`) or [`EAFNOSUPPORT`] (no
/// such address family). No other error is: running out of descriptors or
/// memory, a permission refused or a port taken says nothing about the
/// address.
fn lacks_address(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::AddrNotAvailable
        || EAFNOSUPPORT.is_some_and(|code| error.raw_os_error() == Some(code))
}

/// What binding a listener on a socket address came to ([`bind_outcome`]).
#[derive(Debug)]
enum Bound {
    /// It bound.
    Listener(TcpListener),
    /// It failed with an error [`lacks_address`] accepts: why, as a
    /// capability report names it.
    Missing(String),
    /// It failed with `AddrInUse`: the port is taken on that address.
    PortTaken(io::Error),
}

/// Bind a listener on `at`, read by [`bind_outcome`].
fn bind_on(at: SocketAddr) -> Bound {
    bind_outcome(at, TcpListener::bind(at))
}

/// How [`bind_on`] reads the result of binding a listener on `at` — apart
/// from the bind itself, so that errors no hermetic bind produces are tested
/// too.
///
/// # Panics
///
/// On an error that is neither accepted by [`lacks_address`] nor
/// `AddrInUse`, naming it: reading it as a missing address would quietly
/// turn a policy check into a weaker run.
fn bind_outcome(at: SocketAddr, result: io::Result<TcpListener>) -> Bound {
    match result {
        Ok(listener) => Bound::Listener(listener),
        Err(error) if lacks_address(&error) => {
            Bound::Missing(format!("binding {at} failed: {error}"))
        }
        Err(error) if error.kind() == io::ErrorKind::AddrInUse => Bound::PortTaken(error),
        Err(error) => panic!(
            "binding {at} failed, and not because this host lacks {}: {error:?}",
            bracketed(at.ip())
        ),
    }
}

/// Whether this host binds a listener on `address`: `Ok` when one bound on
/// port 0, `Err` with why not when the bind failed with an error
/// [`lacks_address`] accepts.
///
/// # Panics
///
/// On any other bind error ([`bind_outcome`]) — `AddrInUse` included, which
/// on port 0, where the kernel picks the port, means that no port was left,
/// not that the address is missing.
fn probe_bind(address: IpAddr) -> Result<(), String> {
    let at = SocketAddr::new(address, 0);
    match bind_on(at) {
        Bound::Listener(_) => Ok(()),
        Bound::Missing(why) => Err(why),
        Bound::PortTaken(error) => panic!("binding {at} failed: no port was left: {error:?}"),
    }
}

/// [`probe_bind`] of `address`, once per process: the verdict is kept in
/// `verdict` for every later test, on the assumption that the host's
/// addresses do not change while the tests run. A probe that panics keeps
/// nothing, so every test that asks fails the same loud way.
fn probed_once(address: IpAddr, verdict: &OnceLock<Result<(), String>>) -> Result<(), String> {
    verdict.get_or_init(|| probe_bind(address)).clone()
}

/// [`probe_bind`] of `::1`, the IPv6 loopback, once per process.
fn probe_ipv6_loopback() -> Result<(), String> {
    static VERDICT: OnceLock<Result<(), String>> = OnceLock::new();
    probed_once(V6_LOOPBACK, &VERDICT)
}

/// [`probe_bind`] of `127.0.0.2`, a second IPv4 loopback address, once per
/// process (Linux answers for all of `127.0.0.0/8` on `lo`; macOS, by
/// default, only for `127.0.0.1`).
fn probe_second_v4_loopback() -> Result<(), String> {
    static VERDICT: OnceLock<Result<(), String>> = OnceLock::new();
    probed_once(V4_SECOND_LOOPBACK, &VERDICT)
}

/// An address [`same_port_pair_in`] can put its second listener on.
struct SecondLoopbackCandidate {
    /// The address.
    address: IpAddr,
    /// Whether this host binds it at all: `Err` with why not
    /// ([`probe_bind`]).
    probe: fn() -> Result<(), String>,
    /// What a host must have for it to bind, as a capability report names
    /// it.
    capability: &'static str,
}

impl SecondLoopbackCandidate {
    /// How a capability report names it: `IPv6 loopback ([::1])`.
    fn named(&self) -> String {
        format!("{} ({})", self.capability, bracketed(self.address))
    }
}

/// Where [`same_port_pair`] puts its second listener, in order of
/// preference.
const SECOND_LOOPBACKS: [SecondLoopbackCandidate; 2] = [
    SecondLoopbackCandidate {
        address: V6_LOOPBACK,
        probe: probe_ipv6_loopback,
        capability: "IPv6 loopback",
    },
    SecondLoopbackCandidate {
        address: V4_SECOND_LOOPBACK,
        probe: probe_second_v4_loopback,
        capability: "second IPv4 loopback address",
    },
];

/// A second loopback listener on the port of a `127.0.0.1` one (see
/// [`same_port_pair`]): a dial of `address` on that port is counted, so
/// "never dialled" is checked as no connection.
struct SecondLoopback {
    /// Bound to `address`, on the `127.0.0.1` listener's port.
    listener: TcpListener,
    /// `::1` — the other loopback family — or, on a host without an IPv6
    /// loopback, `127.0.0.2`.
    address: IpAddr,
}

impl SecondLoopback {
    /// Serve `script` on it: the server, and the socket address it listens
    /// on as the assertions name it (`[::1]:<port>`, `127.0.0.2:<port>`).
    fn serve(
        self,
        script: impl Fn(&str) -> Reply + Send + Sync + 'static,
    ) -> (TestServer, SocketAddr) {
        let at = self.listener.local_addr().expect("address");
        (TestServer::on(self.listener, script), at)
    }
}

/// A `127.0.0.1` listener and a second loopback listener on the same port —
/// `[::1]`, the other loopback family, on a host with an IPv6 loopback, and
/// `127.0.0.2` on a host without one — so that "a second address on the
/// same port is never dialled" is checked wherever either binds: the first
/// of [`SECOND_LOOPBACKS`] that binds, as [`same_port_pair_in`] picks it.
fn same_port_pair(test: &str) -> (TcpListener, Option<SecondLoopback>) {
    same_port_pair_in(&SECOND_LOOPBACKS, test)
}

/// A `127.0.0.1` listener and a listener on the first of `candidates`, in
/// order, that binds on the same port.
///
/// A candidate is passed over when its probe, or its bind on the
/// `127.0.0.1` listener's port ([`same_port_on`]), fails with an error
/// [`lacks_address`] accepts; any other bind error panics
/// ([`bind_outcome`]). Each candidate passed over on the way to one that
/// binds gets a `NOTE (capability)` line naming `test`, the listener used
/// instead and the bind error, so that a downgrade is visible under
/// `--nocapture`. When none binds (or there is none), the second listener
/// is `None`, after one `SKIPPED (capability)` line naming `test` and each
/// candidate's bind error, beside a fresh `127.0.0.1` listener: the test
/// then runs every `127.0.0.1` check and skips only those of the second
/// listener.
fn same_port_pair_in(
    candidates: &[SecondLoopbackCandidate],
    test: &str,
) -> (TcpListener, Option<SecondLoopback>) {
    let mut passed_over = Vec::new();
    for candidate in candidates {
        match (candidate.probe)().and_then(|()| same_port_on(candidate.address)) {
            Ok((v4, listener)) => {
                let at = listener.local_addr().expect("address");
                for (named, why) in &passed_over {
                    eprintln!(
                        "NOTE (capability): {test}: {named} is not bindable on this host; \
                         using {at} as the second listener ({why})"
                    );
                }
                let address = candidate.address;
                return (v4, Some(SecondLoopback { listener, address }));
            }
            Err(why) => passed_over.push((candidate.named(), why)),
        }
    }
    let why = if passed_over.is_empty() {
        "there is no address to try".to_string()
    } else {
        passed_over
            .iter()
            .map(|(named, why)| format!("{named}: {why}"))
            .collect::<Vec<_>>()
            .join("; ")
    };
    eprintln!(
        "SKIPPED (capability): {test}: no second loopback listener can share the 127.0.0.1 \
         listener's port ({why}); only its never-dialled checks are skipped, every 127.0.0.1 \
         check still runs"
    );
    (
        TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1"),
        None,
    )
}

/// A `127.0.0.1` listener and an `address` listener on the same port, tried
/// on at most 64 ports: a fresh one each time binding `address` fails with
/// `AddrInUse` (that port is taken there), and a panic when all 64 are.
/// `Err` with why when the bind fails with an error [`lacks_address`]
/// accepts; any other bind error panics ([`bind_outcome`]).
fn same_port_on(address: IpAddr) -> Result<(TcpListener, TcpListener), String> {
    let mut tries = 0;
    loop {
        tries += 1;
        let v4 = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1");
        let port = v4.local_addr().expect("address").port();
        match bind_on(SocketAddr::new(address, port)) {
            Bound::Listener(second) => return Ok((v4, second)),
            Bound::Missing(why) => return Err(why),
            Bound::PortTaken(error) if tries == 64 => panic!(
                "no port is free on both 127.0.0.1 and {} in 64 tries; the last: {error}",
                bracketed(address)
            ),
            Bound::PortTaken(_) => {}
        }
    }
}

/// `address` as a URL's authority writes it: an IPv6 address bracketed.
fn bracketed(address: IpAddr) -> String {
    match address {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => format!("[{v6}]"),
    }
}

/// The address a rebinding or mixed answer names beside `127.0.0.1`: the
/// second listener's, so that a dial of it would be counted — or, on a host
/// that binds no second loopback address, `::1`, which nothing listens on
/// there. Every one of them is refused as loopback, under the real policy
/// and under [`loopback_v4_is_public`] alike.
fn second_answer(second: Option<&SecondLoopback>) -> IpAddr {
    second.map_or(V6_LOOPBACK, |second| second.address)
}

fn code(result: Result<Vec<u8>, ImageInputError>) -> String {
    match result {
        Ok(bytes) => format!("ok ({} bytes)", bytes.len()),
        Err(error) => error.code().to_string(),
    }
}

// ── The happy path, and what a request looks like ───────────────────────

/// An allowlisted loopback listener serves the fixture: the bytes arrive
/// untouched, and resolved through a policy carrying the fetcher they decode
/// to exactly the pixels of the same image sent as a data URI.
#[test]
fn an_allowlisted_image_is_fetched_and_decodes_like_its_data_uri() {
    let image = png();
    let body = image.clone();
    let server = TestServer::start("127.0.0.1", move |_| Reply::Ok(body.clone()));
    let url = server.url("/img.png");
    let fetched = fetcher(5_000, &allow_v4(&server))
        .fetch(&url, 1 << 20)
        .expect("fetched");
    assert_eq!(fetched, image);

    let policy = ImageSourcePolicy::server(None, true)
        .with_remote_fetcher(fetcher(5_000, &allow_v4(&server)).into_shared());
    let via_url = load_image_source(&url, &policy).expect("decoded");
    let via_data_uri =
        load_image_source(PATTERN_PNG_DATA_URI, &ImageSourcePolicy::local_user()).expect("decoded");
    assert_eq!(via_url, via_data_uri, "indistinguishable from the data URI");
    assert_eq!(server.accepts(), 2);
}

/// One `GET` with no body, no cookie, no credentials and the fixed
/// `User-Agent`; the fragment never leaves the process.
#[test]
fn a_fetch_sends_a_bare_get_with_the_fixed_user_agent() {
    let body = png();
    let server = TestServer::start("127.0.0.1", move |_| Reply::Ok(body.clone()));
    let url = server.url("/dir/img.png?size=2#fragment-never-sent");
    fetcher(5_000, &allow_v4(&server))
        .fetch(&url, 1 << 20)
        .expect("fetched");
    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    let head = requests[0].to_ascii_lowercase();
    assert!(
        head.starts_with("get /dir/img.png?size=2 http/1.1\r\n"),
        "{head}"
    );
    assert!(!head.contains("fragment"), "{head}");
    assert!(
        head.contains(&format!(
            "user-agent: oxibonsai/{}\r\n",
            env!("CARGO_PKG_VERSION")
        )),
        "{head}"
    );
    assert!(
        head.contains(&format!("host: 127.0.0.1:{}\r\n", server.port())),
        "{head}"
    );
    for absent in [
        "cookie:",
        "authorization:",
        "proxy-authorization:",
        "content-length:",
        "transfer-encoding:",
        "accept-encoding:",
    ] {
        assert!(!head.contains(absent), "{absent} must not be sent: {head}");
    }
}

// ── S1: without the opt-in nothing is opened ─────────────────────────────

/// No opt-in, or an opt-in with no fetcher installed: the typed refusal,
/// and the listener never sees a connection.
#[test]
fn without_the_opt_in_no_connection_is_made() {
    let _env = test_env::lock();
    let _unset = EnvVarGuard::remove(ALLOW_IMAGE_URL_FETCH_ENV);
    let body = png();
    let server = TestServer::start("127.0.0.1", move |_| Reply::Ok(body.clone()));
    let url = server.url("/img.png");
    #[cfg_attr(not(feature = "server"), allow(unused_mut))]
    let mut policies = vec![
        ImageSourceFlags::default().cli_policy().expect("policy"),
        ImageSourcePolicy::server(None, false),
        ImageSourcePolicy::server(None, true),
    ];
    #[cfg(feature = "server")]
    policies.push(ImageSourceFlags::default().server_policy().expect("policy"));
    for policy in &policies {
        let error = load_image_bytes(&url, policy).expect_err("refused");
        assert_eq!(error.code(), "image_url_fetch_disabled", "{error}");
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(server.accepts(), 0, "nothing was opened");
}

// ── S2: URL syntax ───────────────────────────────────────────────────────

#[test]
fn a_url_the_syntax_rules_refuse_opens_nothing() {
    let server = TestServer::start("127.0.0.1", |_| Reply::Ok(Vec::new()));
    let port = server.port();
    let fetcher = fetcher(5_000, &allow_v4(&server));
    let too_long = format!(
        "http://127.0.0.1:{port}/{}",
        "a".repeat(MAX_REMOTE_IMAGE_URL_BYTES)
    );
    for url in [
        format!("ftp://127.0.0.1:{port}/img.png"),
        format!("file://127.0.0.1:{port}/img.png"),
        format!("http://user:secret@127.0.0.1:{port}/img.png"),
        format!("http://user@127.0.0.1:{port}/img.png"),
        "http:///img.png".to_string(),
        "http://".to_string(),
        format!("http://127.0.0.1:{port}\\@example.com/"),
        format!("http://127.0.0.1:{port}/a b.png"),
        too_long,
    ] {
        let error = fetcher.fetch(&url, 1 << 20).expect_err(&url);
        assert_eq!(error.code(), "image_url_refused", "{url}: {error}");
        assert!(
            !error.to_string().contains("secret"),
            "credentials are never echoed: {error}"
        );
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(server.accepts(), 0);
}

// ── S3: the address policy on every spelling of a literal ────────────────

/// Every spelling of a loopback (or other refused) address is refused
/// before any connection or resolution. Each is checked by its error code
/// (one the policy let through would be dialled, or resolved, instead of
/// refused), and the stub resolver must never be asked. The listeners on the
/// one port every URL names count what reaches them, which must be nothing:
///
/// * `127.0.0.1` counts a dial of any spelling of `127.0.0.1`: the IPv4
///   ones, and the IPv4-mapped `[::ffff:127.0.0.1]` and `[::FFFF:7F00:1]`,
///   which an `AF_INET6` socket would carry to it over IPv4 whether or not
///   the host has `::1`.
/// * The second listener of [`same_port_pair`] counts a dial of its own
///   address. On a host with an IPv6 loopback that is `[::1]`, which only
///   `[::1]` and `[0:0:0:0:0:0:0:1]` denote; on a host without one it is
///   `127.0.0.2`, which every spelling appended from
///   [`V4_SECOND_LOOPBACK_SPELLINGS`] denotes (checked first, through the
///   URL parser) — so there the second listener is a sentinel too, not a
///   bystander.
///
/// The other IPv6 spellings — IPv4-compatible, zoned link-local, unique
/// local — denote addresses nothing here listens on, and the error code
/// alone checks them; so it does the `::1` spellings beside a `127.0.0.2`
/// second listener, and the `127.0.0.2` ones beside `[::1]`. On every host,
/// the IPv4 spellings of `127.0.0.2` hold the loopback rule to all of
/// `127.0.0.0/8`, not just to `127.0.0.1`.
#[test]
fn every_spelling_of_a_refused_literal_is_refused_before_any_connection() {
    let (v4, second) =
        same_port_pair("every_spelling_of_a_refused_literal_is_refused_before_any_connection");
    let port = v4.local_addr().expect("address").port();
    for host in V4_SECOND_LOOPBACK_SPELLINGS {
        let url = format!("http://{host}:{port}/img.png");
        let parsed =
            parse_remote_image_url(&url).unwrap_or_else(|refusal| panic!("{url}: {refusal}"));
        assert_eq!(
            parsed.host().ip().map(|address| address.to_canonical()),
            Some(V4_SECOND_LOOPBACK),
            "{host} denotes 127.0.0.2"
        );
    }
    let v4 = TestServer::on(v4, |_| Reply::Ok(Vec::new()));
    let second = second.map(|second| second.serve(|_| Reply::Ok(Vec::new())));
    let lookup = StubLookup::answering(Vec::new());
    let fetcher = stub_fetcher(5_000, &[], &lookup, classify_address);
    for host in [
        "127.0.0.1",
        "127.0.0.1.",
        "2130706433",
        "0x7f.0.0.1",
        "0X7F.0.0.1",
        "0x7f000001",
        "0177.0.0.1",
        "017700000001",
        "127.1",
        "127.0.1",
        "0.0.0.0",
        "0",
        "[::1]",
        "[0:0:0:0:0:0:0:1]",
        "[::ffff:127.0.0.1]",
        "[::FFFF:7F00:1]",
        "[::127.0.0.1]",
        "[fe80::1%25lo0]",
        "localhost",
        "LOCALHOST.",
        "images.localhost",
        "169.254.169.254",
        "10.0.0.1",
        "[fd00::1]",
    ]
    .into_iter()
    .chain(V4_SECOND_LOOPBACK_SPELLINGS)
    {
        let url = format!("http://{host}:{port}/img.png");
        let error = fetcher.fetch(&url, 1 << 20).expect_err(&url);
        assert_eq!(error.code(), "image_url_refused", "{url}: {error}");
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(v4.accepts(), 0, "nothing reached 127.0.0.1:{port}");
    if let Some((second, at)) = &second {
        assert_eq!(second.accepts(), 0, "nothing reached {at}");
    }
    assert_eq!(
        lookup.calls(),
        0,
        "no literal (or localhost) is ever resolved"
    );
}

/// A name that resolves to a refused address is refused whole, without a
/// connection, and the message never names the address it resolved to.
#[test]
fn a_name_that_resolves_to_a_refused_address_is_refused_without_disclosing_it() {
    let server = TestServer::start("127.0.0.1", |_| Reply::Ok(Vec::new()));
    let port = server.port();
    for answer in [
        vec![V4_LOOPBACK],
        vec![IpAddr::V4(Ipv4Addr::new(10, 0, 0, 7))],
        vec![IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254))],
        vec![IpAddr::V6(
            "::ffff:8.8.8.8".parse::<Ipv6Addr>().expect("v6"),
        )],
        vec![IpAddr::V6("fd00::7".parse::<Ipv6Addr>().expect("v6"))],
    ] {
        let lookup = StubLookup::answering(vec![answer.clone()]);
        let url = format!("http://images.internal.example:{port}/img.png");
        let error = stub_fetcher(5_000, &[], &lookup, classify_address)
            .fetch(&url, 1 << 20)
            .expect_err("refused");
        assert_eq!(error.code(), "image_url_refused", "{error}");
        let message = error.to_string();
        assert!(message.contains("images.internal.example"), "{message}");
        assert!(message.contains("not public"), "{message}");
        for address in &answer {
            assert!(
                !message.contains(&address.to_string()),
                "the resolved address is never disclosed: {message}"
            );
        }
        assert_eq!(lookup.calls(), 1);
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(server.accepts(), 0);
}

// ── S4: the addresses vetted are the addresses dialled ───────────────────

/// A rebinding name server answers a "public" address first and loopback
/// next. The resolution that is vetted is the one the connection uses: the
/// fetch dials the vetted address, the name is resolved exactly once, and
/// the second answer — which a "resolve, check, let the client resolve
/// again" design would have dialled — is never connected to. The next fetch
/// gets the loopback answer, and refuses it. The loopback answer is the
/// address of the second listener of [`same_port_pair`] (`::1`, or
/// `127.0.0.2` on a host without an IPv6 loopback), on the port the vetted
/// address serves, so a dial of it would be counted; on a host that binds
/// neither, the answer stays `::1` and only the never-dialled checks are
/// skipped.
#[test]
fn the_addresses_vetted_are_the_addresses_dialled() {
    let (v4, second) = same_port_pair("the_addresses_vetted_are_the_addresses_dialled");
    let port = v4.local_addr().expect("address").port();
    let body = png();
    let v4 = TestServer::on(v4, move |_| Reply::Ok(body.clone()));
    let rebound = second_answer(second.as_ref());
    let second = second.map(|second| second.serve(|_| Reply::Ok(b"the rebound target".to_vec())));
    let lookup = StubLookup::answering(vec![vec![V4_LOOPBACK], vec![rebound]]);
    let fetcher = stub_fetcher(5_000, &[], &lookup, loopback_v4_is_public);
    let url = format!("http://rebind.example:{port}/img.png");
    let bytes = fetcher
        .fetch(&url, 1 << 20)
        .expect("the vetted address answers");
    assert_eq!(bytes, png());
    assert_eq!(lookup.calls(), 1, "one resolution per connection");
    assert_eq!(v4.accepts(), 1);
    if let Some((second, at)) = &second {
        assert_eq!(
            second.accepts(),
            0,
            "the rebound address {at} is never dialled"
        );
    }

    let error = fetcher
        .fetch(&url, 1 << 20)
        .expect_err("rebound to loopback");
    assert_eq!(error.code(), "image_url_refused", "{error}");
    assert_eq!(lookup.calls(), 2);
    std::thread::sleep(Duration::from_millis(50));
    if let Some((second, at)) = &second {
        assert_eq!(
            second.accepts(),
            0,
            "the rebound address {at} is still never dialled"
        );
    }
}

/// An answer that mixes a public and a refused address is refused whole —
/// no "try the next record" — under the stand-in classification and under
/// the real one (where the public address is never dialled either: the
/// refusal comes before any connection). The refused address of the first
/// answer is that of the second listener of [`same_port_pair`] (`::1`, or
/// `127.0.0.2` on a host without an IPv6 loopback), on the port of the
/// `127.0.0.1` listener, so a dial of either would be counted; on a host
/// that binds neither, the answer keeps `::1` and only the second
/// listener's check is skipped.
#[test]
fn a_mixed_public_and_private_answer_is_refused_whole() {
    let (v4, second) = same_port_pair("a_mixed_public_and_private_answer_is_refused_whole");
    let port = v4.local_addr().expect("address").port();
    let v4 = TestServer::on(v4, |_| Reply::Ok(Vec::new()));
    let private = second_answer(second.as_ref());
    let second = second.map(|second| second.serve(|_| Reply::Ok(Vec::new())));
    let url = format!("http://mixed.example:{port}/img.png");

    let lookup = StubLookup::answering(vec![vec![V4_LOOPBACK, private]]);
    let error = stub_fetcher(5_000, &[], &lookup, loopback_v4_is_public)
        .fetch(&url, 1 << 20)
        .expect_err("refused");
    assert_eq!(error.code(), "image_url_refused", "{error}");

    let public = IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8));
    let lookup = StubLookup::answering(vec![vec![public, V4_LOOPBACK]]);
    let started = Instant::now();
    let error = stub_fetcher(5_000, &[], &lookup, classify_address)
        .fetch(&url, 1 << 20)
        .expect_err("refused");
    assert_eq!(error.code(), "image_url_refused", "{error}");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "refused before any connection attempt"
    );
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(v4.accepts(), 0);
    if let Some((second, at)) = &second {
        assert_eq!(second.accepts(), 0, "nothing reached {at}");
    }
}

// ── S5: redirects ────────────────────────────────────────────────────────

/// A redirect hop is vetted like the first: a hop to a loopback port that
/// is not allowlisted, or to the cloud metadata address, is refused and
/// never connected to.
#[test]
fn a_redirect_to_a_refused_address_is_refused() {
    let target = TestServer::start("127.0.0.1", |_| Reply::Ok(b"internal".to_vec()));
    let target_url = target.url("/internal.png");
    let start = TestServer::start("127.0.0.1", move |head| {
        if head.starts_with("GET /to-loopback ") {
            Reply::Redirect(302, target_url.clone())
        } else {
            Reply::Redirect(302, "http://169.254.169.254/latest/meta-data/".to_string())
        }
    });
    // Only the first hop's host and port are allowlisted.
    let fetcher = fetcher(5_000, &allow_v4(&start));
    for path in ["/to-loopback", "/to-metadata"] {
        let started = Instant::now();
        let error = fetcher.fetch(&start.url(path), 1 << 20).expect_err(path);
        assert_eq!(error.code(), "image_url_refused", "{path}: {error}");
        assert!(error.to_string().contains("after 1 redirect"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(2), "{path}");
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(target.accepts(), 0, "the redirect target was never dialled");
    assert_eq!(start.accepts(), 2);
}

/// `https` → `http` is never followed; `http` → `https` and same-scheme
/// hops are.
#[test]
fn a_redirect_from_https_to_http_is_refused() {
    let secure = parse_remote_image_url("https://images.example/a/b.png").expect("url");
    let refused = redirect_target(&secure, "http://images.example/b.png", 1).expect_err("http");
    assert_eq!(
        refused,
        FetchError::Refused {
            hop: 1,
            refusal: RemoteUrlRefusal::Downgrade
        }
    );
    assert!(redirect_target(&secure, "//other.example/c.png", 1).is_ok());
    assert!(redirect_target(&secure, "c.png", 1).is_ok());
    let plain = parse_remote_image_url("http://images.example/a/b.png").expect("url");
    assert!(redirect_target(&plain, "https://images.example/b.png", 1).is_ok());
    let error = FetchError::Refused {
        hop: 1,
        refusal: RemoteUrlRefusal::Downgrade,
    }
    .into_error("https://images.example/a/b.png");
    assert_eq!(error.code(), "image_url_refused");
}

/// A request goes out only when the HTTP client's own URI parser reads the
/// target as the scheme, host and port that were vetted: a target naming
/// another host, port or scheme, or carrying credentials, is refused.
#[test]
fn the_client_must_read_the_target_as_the_vetted_host() {
    let vetted = parse_remote_image_url("http://images.example:8080/a.png?size=2").expect("url");
    assert!(dialled_is_vetted(&vetted.request_target(), &vetted));
    for target in [
        "http://other.example:8080/a.png",
        "http://images.example:8081/a.png",
        "http://images.example/a.png",
        "https://images.example:8080/a.png",
        "http://user@images.example:8080/a.png",
        "http://images.example:8080@other.example/a.png",
        "not a target",
    ] {
        assert!(!dialled_is_vetted(target, &vetted), "{target}");
    }
    for url in [
        "https://[2001:4860::8888]/a.png",
        "http://93.184.215.14:8080/a.png",
    ] {
        let vetted = parse_remote_image_url(url).expect("url");
        assert!(
            dialled_is_vetted(&vetted.request_target(), &vetted),
            "{url}"
        );
    }
}

/// A redirect loop ends at the cap with a typed error: the first request
/// and three redirects, never a fifth connection.
#[test]
fn a_redirect_loop_ends_at_the_cap() {
    let server = TestServer::start("127.0.0.1", |_| Reply::Redirect(302, "/loop".to_string()));
    let error = fetcher(5_000, &allow_v4(&server))
        .fetch(&server.url("/start"), 1 << 20)
        .expect_err("loop");
    assert_eq!(error.code(), "image_url_fetch_failed", "{error}");
    assert!(
        error.to_string().contains("more than 3 redirects"),
        "{error}"
    );
    assert_eq!(server.accepts(), 1 + MAX_REMOTE_IMAGE_REDIRECTS);
}

/// A relative `Location` resolves against the hop it came from; a
/// redirect's body is never read (one that declares 100 MB and never sends
/// it does not hold the fetch); a redirect without `Location` is a typed
/// failure.
#[test]
fn redirects_resolve_relative_locations_and_skip_their_bodies() {
    let body = png();
    let server = TestServer::start("127.0.0.1", move |head| {
        if head.starts_with("GET /a/b/start ") {
            Reply::RedirectWithEndlessBody(301, "../img.png".to_string())
        } else if head.starts_with("GET /a/img.png ") {
            Reply::Redirect(307, "/final.png?v=1".to_string())
        } else if head.starts_with("GET /final.png?v=1 ") {
            Reply::Ok(body.clone())
        } else {
            Reply::RedirectWithoutLocation(302)
        }
    });
    let fetcher = fetcher(5_000, &allow_v4(&server));
    let started = Instant::now();
    let bytes = fetcher
        .fetch(&server.url("/a/b/start"), 1 << 20)
        .expect("followed");
    assert_eq!(bytes, png());
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "the redirect's endless body was not awaited"
    );
    let paths: Vec<String> = server
        .requests()
        .iter()
        .map(|head| head.lines().next().unwrap_or_default().to_string())
        .collect();
    assert_eq!(
        paths,
        [
            "GET /a/b/start HTTP/1.1",
            "GET /a/img.png HTTP/1.1",
            "GET /final.png?v=1 HTTP/1.1"
        ]
    );

    let error = fetcher
        .fetch(&server.url("/nowhere"), 1 << 20)
        .expect_err("no Location");
    assert_eq!(error.code(), "image_url_fetch_failed", "{error}");
    assert!(error.to_string().contains("without a Location"), "{error}");
}

// ── S7: bounds ───────────────────────────────────────────────────────────

/// Any status but 200 is a failure naming the status; the body the server
/// sent with it is never read into the message, and nothing is retried.
#[test]
fn a_status_other_than_200_is_named_and_its_body_never_echoed() {
    for status in [404u16, 500, 204, 206] {
        let server = TestServer::start("127.0.0.1", move |_| {
            Reply::Status(status, b"SECRET-INTERNAL-ERROR-PAGE".to_vec())
        });
        let error = fetcher(5_000, &allow_v4(&server))
            .fetch(&server.url("/img.png"), 1 << 20)
            .expect_err("not 200");
        assert_eq!(error.code(), "image_url_fetch_failed", "{status}: {error}");
        let message = error.to_string();
        assert!(message.contains(&format!("status {status}")), "{message}");
        assert!(!message.contains("SECRET"), "{message}");
        assert_eq!(server.accepts(), 1, "no retry");
    }
}

/// A `Content-Length` over the cap is refused from the head (the server
/// that declared it never sends a byte, and the fetch does not wait for
/// one); a body with no length, or chunked, is cut off one byte past the
/// cap; a declared length the server overruns is held to the declared length
/// by the HTTP framing, so it cannot pass the cap either.
#[test]
fn the_body_is_held_to_the_cap_whatever_the_server_declares() {
    let cap = 100;
    let declared = TestServer::start("127.0.0.1", |_| Reply::Declared {
        declared: 1_000,
        body: Vec::new(),
    });
    let started = Instant::now();
    let error = fetcher(5_000, &allow_v4(&declared))
        .fetch(&declared.url("/big.png"), cap)
        .expect_err("over the cap");
    assert_eq!(
        error,
        ImageInputError::EncodedTooLarge {
            bytes: 1_000,
            limit: cap
        }
    );
    assert_eq!(error.code(), "image_too_large");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "refused from the head, not after waiting for the body"
    );

    for reply in [
        Reply::Chunked {
            total: 1_000,
            chunk: 64,
        },
        Reply::Unframed { total: 1_000 },
    ] {
        let server = TestServer::start("127.0.0.1", move |_| reply.clone());
        let error = fetcher(5_000, &allow_v4(&server))
            .fetch(&server.url("/stream.png"), cap)
            .expect_err("over the cap");
        assert_eq!(
            error,
            ImageInputError::EncodedTooLarge {
                bytes: cap + 1,
                limit: cap
            }
        );
    }

    // Exactly at the cap is accepted, framed either way.
    for reply in [
        Reply::Chunked {
            total: cap,
            chunk: 30,
        },
        Reply::Unframed { total: cap },
    ] {
        let server = TestServer::start("127.0.0.1", move |_| reply.clone());
        let bytes = fetcher(5_000, &allow_v4(&server))
            .fetch(&server.url("/fits.png"), cap)
            .expect("at the cap");
        assert_eq!(bytes.len(), cap);
    }

    // The server declares 10 bytes and sends 1000: the framing ends the
    // body at 10.
    let lying = TestServer::start("127.0.0.1", |_| Reply::Declared {
        declared: 10,
        body: vec![b'q'; 1_000],
    });
    let bytes = fetcher(5_000, &allow_v4(&lying))
        .fetch(&lying.url("/lying.png"), cap)
        .expect("framed");
    assert_eq!(bytes.len(), 10);
}

/// One deadline covers the whole fetch: a body that drips one byte at a
/// time, or a server that never answers, ends at the deadline — not when
/// the server is done — and the failure names it.
#[test]
fn one_deadline_bounds_the_whole_fetch() {
    for reply in [
        Reply::Drip {
            total: 1_000,
            every: Duration::from_millis(40),
        },
        Reply::Stall(Duration::from_secs(10)),
    ] {
        let server = TestServer::start("127.0.0.1", move |_| reply.clone());
        let started = Instant::now();
        let error = fetcher(300, &allow_v4(&server))
            .fetch(&server.url("/slow.png"), 1 << 20)
            .expect_err("too slow");
        let elapsed = started.elapsed();
        assert_eq!(error.code(), "image_url_fetch_failed", "{error}");
        assert!(
            error.to_string().contains("timed out after 300 ms"),
            "{error}"
        );
        assert!(
            elapsed >= Duration::from_millis(250) && elapsed < Duration::from_millis(1_500),
            "{elapsed:?}"
        );
        // The fetch itself stopped at the deadline — the server sees the
        // connection closed — not just the caller's wait.
        assert!(
            server.wait_finished(1, Duration::from_millis(1_000)),
            "the connection outlived the deadline"
        );
    }
}

/// A connection that is refused, a name that does not resolve and a reply
/// that is not HTTP are failures naming their step.
#[test]
fn transport_failures_name_their_step() {
    // A port nothing listens on.
    let closed = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = closed.local_addr().expect("address").port();
    drop(closed);
    let allow = vec![format!("127.0.0.1:{port}")];
    let error = fetcher(2_000, &allow)
        .fetch(&format!("http://127.0.0.1:{port}/img.png"), 1 << 20)
        .expect_err("refused connection");
    assert_eq!(error.code(), "image_url_fetch_failed");
    assert!(error.to_string().contains("connect"), "{error}");

    let lookup = StubLookup::answering(vec![Vec::new()]);
    let error = stub_fetcher(2_000, &[], &lookup, classify_address)
        .fetch("http://nowhere.example/img.png", 1 << 20)
        .expect_err("no address");
    assert_eq!(error.code(), "image_url_fetch_failed");
    assert!(error.to_string().contains("resolve"), "{error}");

    // A listener that answers with something that is not HTTP.
    let garbage = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = garbage.local_addr().expect("address").port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = garbage.accept() {
            use std::io::Write as _;
            let _ = stream.write_all(b"NOT HTTP AT ALL\r\n\r\n");
        }
    });
    let allow = vec![format!("127.0.0.1:{port}")];
    let error = fetcher(2_000, &allow)
        .fetch(&format!("http://127.0.0.1:{port}/img.png"), 1 << 20)
        .expect_err("not HTTP");
    assert_eq!(error.code(), "image_url_fetch_failed");
    assert!(error.to_string().contains("response"), "{error}");
}

// ── S8: the allowlist ────────────────────────────────────────────────────

#[test]
fn the_allowlist_matches_host_and_port_exactly() {
    let body = png();
    let server = TestServer::start("127.0.0.1", move |_| Reply::Ok(body.clone()));
    let port = server.port();
    let url = server.url("/img.png");
    for allow in [
        "127.0.0.1".to_string(),
        format!("127.0.0.1:{port}"),
        format!("0x7f.0.0.1:{port}"),
    ] {
        fetcher(5_000, std::slice::from_ref(&allow))
            .fetch(&url, 1 << 20)
            .unwrap_or_else(|e| panic!("{allow}: {e}"));
    }
    assert_eq!(server.accepts(), 3);
    // Another port, another host, or no entry: refused, nothing opened.
    for allow in [
        vec![format!("127.0.0.1:{}", port.wrapping_add(1).max(1))],
        vec!["127.0.0.2".to_string()],
        Vec::new(),
    ] {
        let error = fetcher(5_000, &allow)
            .fetch(&url, 1 << 20)
            .expect_err("refused");
        assert_eq!(error.code(), "image_url_refused", "{allow:?}: {error}");
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(server.accepts(), 3);
}

/// A name is matched case-insensitively — `localhost` included, which only
/// an allowlist entry can reach.
#[test]
fn the_allowlist_matches_names_case_insensitively() {
    let body = png();
    let server = TestServer::start("127.0.0.1", move |_| Reply::Ok(body.clone()));
    let port = server.port();
    let lookup = StubLookup::answering(vec![vec![V4_LOOPBACK]]);
    let fetcher = stub_fetcher(
        5_000,
        &[format!("Images.Intranet:{port}")],
        &lookup,
        classify_address,
    );
    let bytes = fetcher
        .fetch(&format!("http://IMAGES.intranet.:{port}/img.png"), 1 << 20)
        .expect("allowlisted");
    assert_eq!(bytes, png());

    let localhost = ImageUrlFetcher::new(settings(5_000, &[format!("LOCALHOST:{port}")]));
    localhost
        .fetch(&format!("http://localhost:{port}/img.png"), 1 << 20)
        .expect("an allowlisted localhost resolves through the system resolver");
    assert_eq!(server.accepts(), 2);
}

#[test]
fn the_settings_resolve_flag_then_environment_then_default() {
    let hosts = |flag: &[&str], env: Option<&str>| {
        let flag: Vec<String> = flag.iter().map(|s| s.to_string()).collect();
        ImageFetchSettings::resolve(None, None, &flag, env).map(|s| s.allowlist.to_string())
    };
    assert_eq!(hosts(&[], None).expect("none"), "(none)");
    assert_eq!(
        hosts(&["b.example:8080"], Some(" a.example , ,c.example ")).expect("both"),
        "a.example, c.example, b.example:8080",
        "the flag's entries are added to the environment's"
    );
    let error = hosts(&["bad host"], None).expect_err("malformed flag entry");
    assert!(
        error
            .to_string()
            .contains("--image-url-allow-host \"bad host\""),
        "{error}"
    );
    let error = hosts(&[], Some("ok.example,http://x")).expect_err("malformed env entry");
    assert!(
        error
            .to_string()
            .contains("OXI_IMAGE_URL_ALLOW_HOSTS entry \"http://x\""),
        "{error}"
    );

    let timeout = |flag: Option<u64>, env: Option<&str>| {
        ImageFetchSettings::resolve(flag, env, &[], None).map(|s| (s.timeout_ms(), s.timeout_from))
    };
    assert_eq!(
        timeout(None, None).expect("default"),
        (10_000, SettingSource::Default)
    );
    assert_eq!(
        timeout(None, Some("2500")).expect("env"),
        (2_500, SettingSource::Env)
    );
    assert_eq!(
        timeout(Some(700), Some("2500")).expect("flag"),
        (700, SettingSource::Flag)
    );
    for (flag, env, names) in [
        (Some(0), None, "--image-url-timeout-ms"),
        (None, Some("0"), "OXI_IMAGE_URL_TIMEOUT_MS"),
        (None, Some("ten"), "OXI_IMAGE_URL_TIMEOUT_MS"),
        (None, Some("-5"), "OXI_IMAGE_URL_TIMEOUT_MS"),
    ] {
        let error = timeout(flag, env).expect_err("refused");
        assert!(error.to_string().contains(names), "{error}");
    }
}

/// An allowlist (the flag or the environment) or an `--image-url-timeout-ms`
/// without the opt-in configures nothing and is refused at start-up, naming
/// the setting; the deadline's environment fallback is not read without the
/// opt-in, so a stale value of it stops nothing; with the opt-in from either
/// source the fetcher is installed.
#[test]
fn an_allowlist_without_the_opt_in_is_a_startup_error() {
    let _env = test_env::lock();
    let _unset = EnvVarGuard::remove(ALLOW_IMAGE_URL_FETCH_ENV);
    let _no_hosts = EnvVarGuard::remove(ALLOW_HOSTS_ENV);
    let _no_timeout = EnvVarGuard::remove(TIMEOUT_ENV);
    let flags = |allow: bool, hosts: &[&str], timeout: Option<u64>| ImageSourceFlags {
        allow_image_url_fetch: allow,
        media_path: None,
        image_url_timeout_ms: timeout,
        image_url_allow_hosts: hosts.iter().map(|s| s.to_string()).collect(),
    };
    let error = flags(false, &["images.intranet"], None)
        .cli_policy()
        .expect_err("allowlist without opt-in");
    assert!(
        error
            .to_string()
            .contains("--image-url-allow-host has no effect"),
        "{error}"
    );
    let error = flags(false, &[], Some(500))
        .cli_policy()
        .expect_err("deadline without opt-in");
    assert!(
        error
            .to_string()
            .contains("--image-url-timeout-ms has no effect"),
        "{error}"
    );
    {
        let _hosts = EnvVarGuard::set(ALLOW_HOSTS_ENV, "images.intranet");
        let error = flags(false, &[], None)
            .cli_policy()
            .expect_err("environment allowlist without opt-in");
        assert!(
            error
                .to_string()
                .contains("OXI_IMAGE_URL_ALLOW_HOSTS is set"),
            "{error}"
        );
    }
    assert!(!flags(false, &[], None)
        .cli_policy()
        .expect("nothing configured")
        .remote
        .is_opted_in());
    {
        // Read only once opted in: without the opt-in even a value that is
        // not a number is never parsed.
        let _timeout = EnvVarGuard::set(TIMEOUT_ENV, "abc");
        assert!(!flags(false, &[], None)
            .cli_policy()
            .expect("an unread deadline variable stops nothing")
            .remote
            .is_opted_in());
        let _on = EnvVarGuard::set(ALLOW_IMAGE_URL_FETCH_ENV, "1");
        let error = flags(false, &[], None)
            .cli_policy()
            .expect_err("opted in, the deadline variable is read");
        assert!(
            error.to_string().contains("OXI_IMAGE_URL_TIMEOUT_MS"),
            "{error}"
        );
    }

    let installed = flags(true, &["images.intranet"], Some(500))
        .cli_policy()
        .expect("opted in");
    let fetcher = installed.remote.fetcher().expect("a fetcher is installed");
    let described = fetcher.fetcher().describe();
    assert!(
        described.contains("per-image deadline 500 ms"),
        "{described}"
    );
    assert!(described.contains("images.intranet"), "{described}");
    {
        let _on = EnvVarGuard::set(ALLOW_IMAGE_URL_FETCH_ENV, "1");
        let report = flags(false, &[], None)
            .remote_fetch_report()
            .expect("opted in by the environment");
        assert!(
            report.to_string().contains("opted in by the environment"),
            "{report}"
        );
        let error = flags(false, &["x:0"], None)
            .cli_policy()
            .expect_err("malformed entry");
        assert!(error.to_string().contains("\"x:0\""), "{error}");
    }
}

// ── S9: what is disclosed, and what is logged ────────────────────────────

/// Events a test captures on its own thread.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<String>>>);

impl Captured {
    fn lines(&self) -> Vec<String> {
        self.0.lock().map(|lines| lines.clone()).unwrap_or_default()
    }
}

struct FieldText(String);

impl tracing::field::Visit for FieldText {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push_str(&format!("{}={value:?} ", field.name()));
    }
}

impl tracing::Subscriber for Captured {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut text = FieldText(String::new());
        event.record(&mut text);
        if let Ok(mut lines) = self.0.lock() {
            lines.push(text.0);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Set by [`logs_and_messages_never_carry_the_path_the_query_or_a_resolved_address`]
/// for its child process.
const LOG_CHILD_ENV: &str = "OXIBONSAI_TEST_IMAGE_FETCH_LOG_CHILD";

/// Run the test `name` of this module alone, in a child process of this
/// test binary with `env` set: for a test that must own the process — its
/// environment, or its log call sites, which a thread-local capture shares
/// with every concurrently running test.
fn run_child(name: &str, env: &[(&str, String)]) -> std::process::Output {
    let module = module_path!();
    let module = module.split_once("::").map_or(module, |(_, rest)| rest);
    let mut child = std::process::Command::new(std::env::current_exe().expect("this test binary"));
    child.args([
        format!("{module}::{name}").as_str(),
        "--exact",
        "--nocapture",
        "--test-threads",
        "1",
    ]);
    for (key, value) in env {
        child.env(key, value);
    }
    child.output().expect("run the child test")
}

/// Assert that a child run of one test passed and actually ran it.
fn assert_child_ran(output: &std::process::Output) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(
        stdout.contains("1 passed"),
        "the child ran the test: {stdout}\n{stderr}"
    );
}

/// The log lines of a fetch name the scheme, host and port, the status,
/// the byte count and the milliseconds — never the path or the query (they
/// can carry tokens) and never a resolved address; the messages a client
/// sees never name a resolved address either. (Checked in a child process
/// of its own, where no other test's log calls share the call sites the
/// capture depends on.)
#[test]
fn logs_and_messages_never_carry_the_path_the_query_or_a_resolved_address() {
    assert_child_ran(&run_child(
        "log_capture_child",
        &[(LOG_CHILD_ENV, "1".to_string())],
    ));
}

/// The child half of
/// [`logs_and_messages_never_carry_the_path_the_query_or_a_resolved_address`]:
/// a no-op unless that test started this process.
#[test]
fn log_capture_child() {
    if std::env::var(LOG_CHILD_ENV).is_err() {
        return;
    }
    let body = png();
    let server = TestServer::start("127.0.0.1", move |head| {
        if head.starts_with("GET /missing") {
            Reply::Status(404, b"nope".to_vec())
        } else {
            Reply::Ok(body.clone())
        }
    });
    let port = server.port();
    let lookup = StubLookup::answering(vec![
        vec![V4_LOOPBACK],
        vec![V4_LOOPBACK],
        vec![IpAddr::V4(Ipv4Addr::new(10, 9, 8, 7))],
    ]);
    let fetcher = stub_fetcher(
        5_000,
        &[format!("images.example:{port}")],
        &lookup,
        classify_address,
    );
    let refusing = stub_fetcher(5_000, &[], &lookup, classify_address);
    let captured = Captured::default();
    let (ok, missing, refused) = tracing::subscriber::with_default(captured.clone(), || {
        // A log call site another test hit first (with no subscriber that
        // wanted it) may have cached "never"; recompute every call site's
        // interest now that this thread's capture is registered.
        tracing::callsite::rebuild_interest_cache();
        (
            fetcher.fetch(
                &format!("http://images.example:{port}/secret-path/img.png?token=s3cr3t"),
                1 << 20,
            ),
            fetcher.fetch(
                &format!("http://images.example:{port}/missing?token=s3cr3t"),
                1 << 20,
            ),
            refusing.fetch(
                &format!("http://images.example:{port}/x.png?token=s3cr3t"),
                1 << 20,
            ),
        )
    });
    assert_eq!(code(ok), format!("ok ({} bytes)", png().len()));
    let missing = missing.expect_err("404");
    let refused = refused.expect_err("10.9.8.7 is private");
    assert_eq!(refused.code(), "image_url_refused");
    assert!(!refused.to_string().contains("10.9.8.7"), "{refused}");
    assert!(!missing.to_string().contains("127.0.0.1"), "{missing}");

    let lines = captured.lines();
    assert!(lines.len() >= 3, "{lines:?}");
    let all = lines.join("\n");
    assert!(
        all.contains(&format!("http://images.example:{port}")),
        "{all}"
    );
    assert!(all.contains("status=200"), "{all}");
    assert!(all.contains("status=404"), "{all}");
    assert!(all.contains(&format!("bytes={}", png().len())), "{all}");
    assert!(all.contains("ms="), "{all}");
    for secret in [
        "secret-path",
        "token",
        "s3cr3t",
        "/missing",
        "10.9.8.7",
        "127.0.0.1",
    ] {
        assert!(!all.contains(secret), "{secret} leaked into the log: {all}");
    }
}

// ── S6: no proxy ─────────────────────────────────────────────────────────

/// Set by [`the_proxy_environment_is_never_used`] for its child process:
/// the image URL [`proxy_child`] fetches.
const PROXY_CHILD_URL_ENV: &str = "OXIBONSAI_TEST_IMAGE_FETCH_CHILD_URL";
/// The allowlist entry the child uses.
const PROXY_CHILD_ALLOW_ENV: &str = "OXIBONSAI_TEST_IMAGE_FETCH_CHILD_ALLOW";

/// The child half of [`the_proxy_environment_is_never_used`]: a no-op
/// unless that test started this process with its environment.
#[test]
fn proxy_child() {
    let (Ok(url), Ok(allow)) = (
        std::env::var(PROXY_CHILD_URL_ENV),
        std::env::var(PROXY_CHILD_ALLOW_ENV),
    ) else {
        return;
    };
    let bytes = fetcher(5_000, &[allow])
        .fetch(&url, 1 << 20)
        .expect("fetched directly");
    assert_eq!(bytes, png());
}

/// The fetch never goes through a proxy, whatever the environment says:
/// a child process with `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` (both
/// cases) pointing at a listener fetches the image directly, and the proxy
/// listener sees no connection. (A child process, because the environment
/// is shared by every test thread of this one.)
#[test]
fn the_proxy_environment_is_never_used() {
    let proxy = TestServer::start("127.0.0.1", |_| Reply::Status(502, b"proxy".to_vec()));
    let body = png();
    let image = TestServer::start("127.0.0.1", move |_| Reply::Ok(body.clone()));
    let proxy_url = format!("http://127.0.0.1:{}", proxy.port());
    let module = module_path!();
    let module = module.split_once("::").map_or(module, |(_, rest)| rest);
    let mut child = std::process::Command::new(std::env::current_exe().expect("this test binary"));
    child
        .args([
            format!("{module}::proxy_child").as_str(),
            "--exact",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(PROXY_CHILD_URL_ENV, image.url("/img.png"))
        .env(PROXY_CHILD_ALLOW_ENV, format!("127.0.0.1:{}", image.port()))
        .env_remove("NO_PROXY")
        .env_remove("no_proxy");
    for variable in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ] {
        child.env(variable, &proxy_url);
    }
    let output = child.output().expect("run the child test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(
        stdout.contains("1 passed"),
        "the child ran the fetch: {stdout}\n{stderr}"
    );
    assert_eq!(image.accepts(), 1, "fetched directly");
    assert_eq!(proxy.accepts(), 0, "the proxy saw nothing");
}

// ── The bridge: every thread a fetch is called from ──────────────────────

fn fetch_fixture() -> (TestServer, ImageUrlFetcher) {
    let body = png();
    let server = TestServer::start("127.0.0.1", move |_| Reply::Ok(body.clone()));
    let fetcher = fetcher(5_000, &allow_v4(&server));
    (server, fetcher)
}

#[test]
fn a_fetch_works_from_a_plain_thread() {
    let (server, fetcher) = fetch_fixture();
    let url = server.url("/img.png");
    let bytes = std::thread::spawn(move || fetcher.fetch(&url, 1 << 20))
        .join()
        .expect("the thread finished")
        .expect("fetched");
    assert_eq!(bytes, png());
}

/// Called straight from an async task on a multi-thread runtime worker
/// (where a nested `block_on` panics): no panic, the image arrives.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fetch_works_from_inside_a_multi_thread_runtime_worker() {
    let (server, fetcher) = fetch_fixture();
    let bytes = fetcher
        .fetch(&server.url("/img.png"), 1 << 20)
        .expect("fetched");
    assert_eq!(bytes, png());
}

/// Called from the only thread of a current-thread runtime (where waiting
/// on that runtime for the answer would deadlock): the fetcher's own runtime
/// answers.
#[tokio::test(flavor = "current_thread")]
async fn a_fetch_works_from_a_current_thread_runtime() {
    let (server, fetcher) = fetch_fixture();
    let bytes = fetcher
        .fetch(&server.url("/img.png"), 1 << 20)
        .expect("fetched");
    assert_eq!(bytes, png());
}

/// The fetcher starts its thread on its first fetch: an opted-in command
/// that never sees a remote image never starts one, and a preflight check
/// never needs one.
#[test]
fn preflight_needs_no_network_and_no_thread() {
    let server = TestServer::start("127.0.0.1", |_| Reply::Ok(Vec::new()));
    let allowed = fetcher(5_000, &allow_v4(&server));
    allowed
        .preflight(&server.url("/img.png"))
        .expect("an allowlisted literal passes");
    let refused = fetcher(5_000, &[])
        .preflight(&server.url("/img.png"))
        .expect_err("loopback without an allowlist");
    assert_eq!(refused.code(), "image_url_refused");
    assert!(allowed.worker.get().is_none(), "no thread was started");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(server.accepts(), 0);
}

// ── The second listener of `same_port_pair` ─────────────────────────────

/// TEST-NET-1 (RFC 5737), documentation only and on no interface: binding a
/// listener on it fails with `AddrNotAvailable`, which [`lacks_address`]
/// accepts — on a host that lets a process bind only addresses it has
/// (on Linux, `net.ipv4.ip_nonlocal_bind` unset).
const NOT_LOCAL_V4: IpAddr = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1));

/// The IPv6 documentation prefix (RFC 3849), on no interface either:
/// binding it fails with `AddrNotAvailable` where the host has IPv6, and
/// with [`EAFNOSUPPORT`] where its kernel has none (on Linux,
/// `net.ipv6.ip_nonlocal_bind` unset).
const NOT_LOCAL_V6: IpAddr = IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1));

/// Assert that `listener` is a live `127.0.0.1` listener: it accepts a
/// connection to its address, from the client that made it.
fn assert_live_v4_listener(listener: &TcpListener) {
    let at = listener.local_addr().expect("address");
    assert_eq!(at.ip(), V4_LOOPBACK, "a 127.0.0.1 listener: {at}");
    let client = TcpStream::connect(at).expect("connect to the 127.0.0.1 listener");
    let (_, peer) = listener.accept().expect("the 127.0.0.1 listener accepts");
    assert_eq!(peer, client.local_addr().expect("address"), "{at}");
}

/// Only `AddrNotAvailable` and `EAFNOSUPPORT` read as a missing address and
/// only `AddrInUse` as a taken port; running out of descriptors or memory, a
/// permission refused or anything else reads as neither.
#[test]
fn only_a_missing_address_or_family_reads_as_a_missing_address() {
    let mut missing = vec![io::Error::from(io::ErrorKind::AddrNotAvailable)];
    missing.extend(EAFNOSUPPORT.map(io::Error::from_raw_os_error));
    for error in &missing {
        assert!(lacks_address(error), "{error:?}");
    }
    for error in [
        io::Error::from(io::ErrorKind::AddrInUse),
        io::Error::from(io::ErrorKind::PermissionDenied),
        io::Error::from(io::ErrorKind::OutOfMemory),
        // EMFILE, "too many open files", on every Unix.
        io::Error::from_raw_os_error(24),
        io::Error::other("an unexplained failure"),
    ] {
        assert!(!lacks_address(&error), "{error:?}");
    }

    let at = SocketAddr::new(V6_LOOPBACK, 4_242);
    for error in missing {
        match bind_outcome(at, Err(error)) {
            Bound::Missing(why) => {
                assert!(why.starts_with("binding [::1]:4242 failed: "), "{why}");
            }
            other => panic!("not read as a missing address: {other:?}"),
        }
    }
    let taken = bind_outcome(at, Err(io::Error::from(io::ErrorKind::AddrInUse)));
    assert!(matches!(taken, Bound::PortTaken(_)), "{taken:?}");
}

/// A bind error that is neither a missing address nor a taken port is loud:
/// it panics, naming the error, instead of passing a candidate over.
#[test]
#[should_panic(expected = "binding [::1]:0 failed, and not because this host lacks [::1]")]
fn an_unexplained_bind_error_panics() {
    bind_outcome(
        SocketAddr::new(V6_LOOPBACK, 0),
        Err(io::Error::from(io::ErrorKind::PermissionDenied)),
    );
}

/// Candidates the host lacks are passed over without a panic: each probe
/// reads its bind failure as a missing address, the second listener is
/// `None` — after a `SKIPPED (capability)` line naming both, the outcome
/// this test expects — and the `127.0.0.1` listener handed back is live.
#[test]
fn second_addresses_the_host_lacks_are_passed_over_without_a_panic() {
    let candidates = [
        SecondLoopbackCandidate {
            address: NOT_LOCAL_V4,
            probe: || probe_bind(NOT_LOCAL_V4),
            capability: "TEST-NET-1 address",
        },
        SecondLoopbackCandidate {
            address: NOT_LOCAL_V6,
            probe: || probe_bind(NOT_LOCAL_V6),
            capability: "IPv6 documentation address",
        },
    ];
    for candidate in &candidates {
        let probed = (candidate.probe)();
        let failed = format!("binding {} failed: ", SocketAddr::new(candidate.address, 0));
        assert!(
            probed.as_ref().is_err_and(|why| why.starts_with(&failed)),
            "{} is on no interface, so binding it must fail as a missing address (a host that \
             lets a process bind addresses it does not have cannot run this test): {probed:?}",
            candidate.named()
        );
    }
    let (v4, second) = same_port_pair_in(
        &candidates,
        "second_addresses_the_host_lacks_are_passed_over_without_a_panic (an expected skip)",
    );
    assert!(second.is_none(), "an address the host lacks was bound");
    assert_live_v4_listener(&v4);
}

/// No candidate at all: the second listener is `None` — after a `SKIPPED
/// (capability)` line, the outcome this test expects — and the `127.0.0.1`
/// listener handed back is live.
#[test]
fn an_empty_candidate_list_gives_no_second_listener_without_a_panic() {
    let (v4, second) = same_port_pair_in(
        &[],
        "an_empty_candidate_list_gives_no_second_listener_without_a_panic (an expected skip)",
    );
    assert!(second.is_none(), "no candidate, yet a second listener");
    assert_live_v4_listener(&v4);
}
