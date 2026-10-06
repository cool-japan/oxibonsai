//! The address policy and the URL rules, as tables.

use super::*;

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::atomic::{AtomicUsize, Ordering};

fn v4(text: &str) -> IpAddr {
    IpAddr::V4(text.parse::<Ipv4Addr>().expect(text))
}

fn v6(text: &str) -> IpAddr {
    IpAddr::V6(text.parse::<Ipv6Addr>().expect(text))
}

// ── Address classes ──────────────────────────────────────────────────────

/// Every refused IPv4 class, at both of its edges, with the public address
/// on either side of it.
#[test]
fn every_refused_ipv4_class_is_refused_at_its_edges_and_nowhere_else() {
    use AddressClass as C;
    let table: &[(&str, Option<AddressClass>)] = &[
        ("0.0.0.0", Some(C::Unspecified)),
        ("0.255.255.255", Some(C::Unspecified)),
        ("1.0.0.0", None),
        ("9.255.255.255", None),
        ("10.0.0.0", Some(C::Private)),
        ("10.255.255.255", Some(C::Private)),
        ("11.0.0.0", None),
        ("100.63.255.255", None),
        ("100.64.0.0", Some(C::SharedAddressSpace)),
        ("100.127.255.255", Some(C::SharedAddressSpace)),
        ("100.128.0.0", None),
        ("126.255.255.255", None),
        ("127.0.0.1", Some(C::Loopback)),
        ("127.255.255.255", Some(C::Loopback)),
        ("128.0.0.0", None),
        ("169.253.255.255", None),
        ("169.254.0.0", Some(C::LinkLocal)),
        ("169.254.169.254", Some(C::LinkLocal)),
        ("169.254.255.255", Some(C::LinkLocal)),
        ("169.255.0.0", None),
        ("172.15.255.255", None),
        ("172.16.0.0", Some(C::Private)),
        ("172.31.255.255", Some(C::Private)),
        ("172.32.0.0", None),
        ("191.255.255.255", None),
        ("192.0.0.0", Some(C::ProtocolAssignments)),
        ("192.0.0.255", Some(C::ProtocolAssignments)),
        ("192.0.1.0", None),
        ("192.0.2.0", Some(C::Documentation)),
        ("192.0.2.255", Some(C::Documentation)),
        ("192.0.3.0", None),
        ("192.88.98.255", None),
        ("192.88.99.0", Some(C::SixToFourRelay)),
        ("192.88.99.255", Some(C::SixToFourRelay)),
        ("192.88.100.0", None),
        ("192.167.255.255", None),
        ("192.168.0.0", Some(C::Private)),
        ("192.168.255.255", Some(C::Private)),
        ("192.169.0.0", None),
        ("198.17.255.255", None),
        ("198.18.0.0", Some(C::Benchmarking)),
        ("198.19.255.255", Some(C::Benchmarking)),
        ("198.20.0.0", None),
        ("198.51.99.255", None),
        ("198.51.100.0", Some(C::Documentation)),
        ("198.51.100.255", Some(C::Documentation)),
        ("198.51.101.0", None),
        ("203.0.112.255", None),
        ("203.0.113.0", Some(C::Documentation)),
        ("203.0.113.255", Some(C::Documentation)),
        ("203.0.114.0", None),
        ("223.255.255.255", None),
        ("224.0.0.0", Some(C::Multicast)),
        ("239.255.255.255", Some(C::Multicast)),
        ("240.0.0.0", Some(C::Reserved)),
        ("255.255.255.254", Some(C::Reserved)),
        ("255.255.255.255", Some(C::Broadcast)),
        ("8.8.8.8", None),
        ("1.1.1.1", None),
        ("93.184.216.34", None),
    ];
    for (text, expected) in table {
        assert_eq!(classify_address(v4(text)), *expected, "{text}");
        assert_eq!(is_public_address(v4(text)), expected.is_none(), "{text}");
    }
}

/// Every refused IPv6 class, at its edges, and the mapped / compatible forms
/// of a public IPv4 address refused all the same.
#[test]
fn every_refused_ipv6_class_is_refused_at_its_edges_and_nowhere_else() {
    use AddressClass as C;
    let table: &[(&str, Option<AddressClass>)] = &[
        ("::", Some(C::Unspecified)),
        ("::1", Some(C::Loopback)),
        ("::2", Some(C::Ipv4Compatible)),
        ("::127.0.0.1", Some(C::Ipv4Compatible)),
        ("::8.8.8.8", Some(C::Ipv4Compatible)),
        ("::ffff:127.0.0.1", Some(C::Ipv4Mapped)),
        ("::ffff:8.8.8.8", Some(C::Ipv4Mapped)),
        ("::ffff:169.254.169.254", Some(C::Ipv4Mapped)),
        ("::1:0:0:0", Some(C::Reserved)),
        ("64:ff9b::808:808", Some(C::Nat64)),
        ("64:ff9b::ffff:ffff", Some(C::Nat64)),
        ("64:ff9b:1::1", Some(C::Nat64)),
        ("64:ff9b:1:ffff::1", Some(C::Nat64)),
        ("64:ff9b:2::1", Some(C::Reserved)),
        ("100::", Some(C::DiscardOnly)),
        ("100::ffff:ffff:ffff:ffff", Some(C::DiscardOnly)),
        ("100:0:0:1::", Some(C::Reserved)),
        ("1fff:ffff::1", Some(C::Reserved)),
        ("2000::1", None),
        ("2001::", Some(C::Teredo)),
        ("2001:0:ffff:ffff::1", Some(C::Teredo)),
        ("2001:1::1", Some(C::ProtocolAssignments)),
        ("2001:2::1", Some(C::Benchmarking)),
        ("2001:2:0:ffff::1", Some(C::Benchmarking)),
        ("2001:2:1::1", Some(C::ProtocolAssignments)),
        ("2001:10::1", Some(C::ProtocolAssignments)),
        ("2001:20::1", Some(C::ProtocolAssignments)),
        ("2001:1ff:ffff::1", Some(C::ProtocolAssignments)),
        ("2001:200::1", None),
        ("2001:db7:ffff::1", None),
        ("2001:db8::", Some(C::Documentation)),
        ("2001:db8:ffff:ffff::1", Some(C::Documentation)),
        ("2001:db9::1", None),
        ("2001:4860:4860::8888", None),
        ("2001:ffff::1", None),
        ("2002::", Some(C::SixToFour)),
        ("2002:7f00:1::1", Some(C::SixToFour)),
        ("2002:ffff::1", Some(C::SixToFour)),
        ("2003::1", None),
        ("2606:4700:4700::1111", None),
        ("2a00::1", None),
        ("3ffe:ffff::1", None),
        ("3fff::1", Some(C::Documentation)),
        ("3fff:fff:ffff::1", Some(C::Documentation)),
        ("3fff:1000::1", None),
        ("4000::1", Some(C::Reserved)),
        ("fbff::1", Some(C::Reserved)),
        ("fc00::", Some(C::UniqueLocal)),
        ("fd12:3456::1", Some(C::UniqueLocal)),
        ("fdff:ffff::1", Some(C::UniqueLocal)),
        ("fe7f::1", Some(C::Reserved)),
        ("fe80::1", Some(C::LinkLocal)),
        ("febf:ffff::1", Some(C::LinkLocal)),
        ("fec0::1", Some(C::SiteLocal)),
        ("feff:ffff::1", Some(C::SiteLocal)),
        ("ff00::", Some(C::Multicast)),
        ("ff02::1", Some(C::Multicast)),
        (
            "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
            Some(C::Multicast),
        ),
    ];
    for (text, expected) in table {
        assert_eq!(classify_address(v6(text)), *expected, "{text}");
    }
}

// ── IP literals in every spelling ────────────────────────────────────────

/// Every spelling of an address the client or the C library resolver
/// accepts reaches the classifier as that address; none is a name.
#[test]
fn every_literal_spelling_is_vetted_as_the_address_it_names() {
    let loopback = v4("127.0.0.1");
    let table: &[(&str, IpAddr)] = &[
        ("http://127.0.0.1/x.png", loopback),
        ("http://2130706433/x.png", loopback),
        ("http://0x7f.0.0.1/x.png", loopback),
        ("http://0X7F.0.0.1/x.png", loopback),
        ("http://0x7F000001/x.png", loopback),
        ("http://0177.0.0.1/x.png", loopback),
        ("http://017700000001/x.png", loopback),
        ("http://127.1/x.png", loopback),
        ("http://127.0.1/x.png", loopback),
        ("http://127.0.0.1./x.png", loopback),
        ("http://0x7f.1:8080/x.png", loopback),
        ("http://0.0.0.0/x.png", v4("0.0.0.0")),
        ("http://0/x.png", v4("0.0.0.0")),
        (
            "http://169.254.169.254/latest/meta-data/",
            v4("169.254.169.254"),
        ),
        ("http://0xa9fea9fe/", v4("169.254.169.254")),
        ("http://[::1]/x.png", v6("::1")),
        ("http://[0:0:0:0:0:0:0:1]/x.png", v6("::1")),
        ("http://[::ffff:127.0.0.1]/x.png", v6("::ffff:127.0.0.1")),
        ("http://[::FFFF:7F00:1]/x.png", v6("::ffff:127.0.0.1")),
        ("http://[FE80::1]:8080/x.png", v6("fe80::1")),
    ];
    let none = HostAllowlist::default();
    for (url, address) in table {
        let parsed = parse_remote_image_url(url).expect(url);
        assert_eq!(parsed.host().ip(), Some(*address), "{url}");
        let refusal = vet_target(&parsed, &none).expect_err(url);
        assert!(
            matches!(refusal, RemoteUrlRefusal::NotPublic { class: Some(_), .. }),
            "{url}: {refusal:?}"
        );
    }
    // The canonical form is what is dialled.
    assert_eq!(
        parse_remote_image_url("http://0x7f.1:8080/x.png")
            .expect("parses")
            .request_target(),
        "http://127.0.0.1:8080/x.png"
    );
    assert_eq!(
        parse_remote_image_url("http://[0:0:0:0:0:0:0:1]/x.png")
            .expect("parses")
            .request_target(),
        "http://[::1]/x.png"
    );
    // A public literal in an odd spelling passes as that address.
    let public = parse_remote_image_url("http://0x08080808/x.png").expect("parses");
    assert_eq!(
        vet_target(&public, &none).expect("public"),
        TargetVerdict::PublicLiteral(v4("8.8.8.8"))
    );
    assert_eq!(public.request_target(), "http://8.8.8.8/x.png");
}

/// A zone identifier, an unbracketed IPv6 address, and a host that ends in
/// a number without being an IPv4 address are refused, never resolved.
#[test]
fn malformed_literals_are_refused_never_treated_as_names() {
    for url in ["http://[fe80::1%25en0]/", "http://[fe80::1%en0]/x"] {
        assert_eq!(
            parse_remote_image_url(url).expect_err(url),
            RemoteUrlRefusal::ZoneId,
            "{url}"
        );
    }
    for url in [
        "http://256.0.0.1/",
        "http://1.2.3.4.5/",
        "http://0x100000000/",
        "http://4294967296/",
        "http://08.0.0.1/",
        "http://127.0.0.256/",
        "http://foo.123/",
        "http://::1/",
        "http://[::1/",
        "http://[::1]x/",
        "http://[not-v6]/",
        "http://%31%32%37.0.0.1/",
        "http://a..b/",
        "http://.example.com/",
        "http://bücher.example/",
        "http://under score!/",
    ] {
        let refusal = parse_remote_image_url(url).expect_err(url);
        assert!(
            matches!(
                refusal,
                RemoteUrlRefusal::InvalidHost { .. } | RemoteUrlRefusal::Malformed { .. }
            ),
            "{url}: {refusal:?}"
        );
    }
}

// ── URL syntax ───────────────────────────────────────────────────────────

/// Recognises one kind of refusal.
type IsRefusal = fn(&RemoteUrlRefusal) -> bool;

#[test]
fn only_http_and_https_with_a_host_and_no_credentials_are_accepted() {
    let ok = parse_remote_image_url("HTTPS://Images.Example.COM./a/b.png?size=2#frag").expect("ok");
    assert_eq!(ok.scheme(), RemoteScheme::Https);
    assert_eq!(
        ok.host(),
        &RemoteHost::Domain("images.example.com".to_string())
    );
    assert_eq!(ok.port(), 443);
    assert_eq!(ok.path(), "/a/b.png");
    assert_eq!(ok.query(), Some("size=2"));
    assert_eq!(
        ok.request_target(),
        "https://images.example.com/a/b.png?size=2",
        "the fragment is never sent"
    );
    assert_eq!(ok.origin(), "https://images.example.com:443");
    assert_eq!(
        parse_remote_image_url("http://h")
            .expect("no path")
            .request_target(),
        "http://h/"
    );

    let refused: &[(&str, IsRefusal)] = &[
        ("ftp://example.com/x.png", |r| {
            matches!(r, RemoteUrlRefusal::Scheme { .. })
        }),
        ("file:///etc/passwd", |r| {
            matches!(r, RemoteUrlRefusal::Scheme { .. })
        }),
        ("gopher://example.com/", |r| {
            matches!(r, RemoteUrlRefusal::Scheme { .. })
        }),
        ("javascript:alert(1)", |r| {
            matches!(r, RemoteUrlRefusal::Malformed { .. })
        }),
        ("http:example.com/x", |r| {
            matches!(r, RemoteUrlRefusal::Malformed { .. })
        }),
        ("http://user:pass@example.com/x.png", |r| {
            *r == RemoteUrlRefusal::Userinfo
        }),
        ("http://user@example.com/x.png", |r| {
            *r == RemoteUrlRefusal::Userinfo
        }),
        ("http://@example.com/", |r| *r == RemoteUrlRefusal::Userinfo),
        ("http:///x.png", |r| *r == RemoteUrlRefusal::EmptyHost),
        ("http://", |r| *r == RemoteUrlRefusal::EmptyHost),
        ("https://?q", |r| *r == RemoteUrlRefusal::EmptyHost),
        ("http://:80/", |r| *r == RemoteUrlRefusal::EmptyHost),
        ("http://example.com:0/", |r| {
            *r == RemoteUrlRefusal::InvalidPort
        }),
        ("http://example.com:65536/", |r| {
            *r == RemoteUrlRefusal::InvalidPort
        }),
        ("http://example.com:/", |r| {
            *r == RemoteUrlRefusal::InvalidPort
        }),
        ("http://example.com:8o/", |r| {
            *r == RemoteUrlRefusal::InvalidPort
        }),
        ("http://example.com/a b.png", |r| {
            matches!(r, RemoteUrlRefusal::Malformed { .. })
        }),
        ("http://example.com/a\tb.png", |r| {
            matches!(r, RemoteUrlRefusal::Malformed { .. })
        }),
        ("http://evil.example\\@127.0.0.1/", |r| {
            matches!(r, RemoteUrlRefusal::Malformed { .. })
        }),
        ("http://example.com/a\\b.png", |r| {
            matches!(r, RemoteUrlRefusal::Malformed { .. })
        }),
    ];
    for (url, expected) in refused {
        let refusal = parse_remote_image_url(url).expect_err(url);
        assert!(expected(&refusal), "{url}: {refusal:?}");
    }
}

#[test]
fn the_length_cap_is_2048_bytes() {
    let base = "http://example.com/";
    let fits = format!(
        "{base}{}",
        "a".repeat(MAX_REMOTE_IMAGE_URL_BYTES - base.len())
    );
    assert_eq!(fits.len(), MAX_REMOTE_IMAGE_URL_BYTES);
    parse_remote_image_url(&fits).expect("exactly the cap");
    let over = format!("{fits}a");
    assert_eq!(
        parse_remote_image_url(&over).expect_err("one past"),
        RemoteUrlRefusal::TooLong {
            bytes: MAX_REMOTE_IMAGE_URL_BYTES + 1
        }
    );
}

#[test]
fn a_request_target_carries_only_bytes_a_request_line_may() {
    let parsed = parse_remote_image_url("http://h/caf\u{e9}/<x>|^`{}\"[]?q=a|b&c=<d>").expect("ok");
    assert_eq!(
        parsed.request_target(),
        "http://h/caf%C3%A9/%3Cx%3E%7C%5E%60%7B%7D%22%5B%5D?q=a%7Cb&c=%3Cd%3E"
    );
    // An existing escape is kept as it is.
    assert_eq!(
        parse_remote_image_url("http://h/a%20b.png")
            .expect("ok")
            .request_target(),
        "http://h/a%20b.png"
    );
}

// ── Redirect targets (RFC 3986 §5.2, the normal examples) ────────────────

#[test]
fn a_location_resolves_against_the_url_it_came_from() {
    let base = parse_remote_image_url("http://a/b/c/d;p?q").expect("base");
    let table: &[(&str, &str)] = &[
        ("g", "http://a/b/c/g"),
        ("./g", "http://a/b/c/g"),
        ("g/", "http://a/b/c/g/"),
        ("/g", "http://a/g"),
        ("//g", "http://g/"),
        ("?y", "http://a/b/c/d;p?y"),
        ("g?y", "http://a/b/c/g?y"),
        ("#s", "http://a/b/c/d;p?q"),
        ("g#s", "http://a/b/c/g"),
        ("g?y#s", "http://a/b/c/g?y"),
        (";x", "http://a/b/c/;x"),
        ("g;x", "http://a/b/c/g;x"),
        (".", "http://a/b/c/"),
        ("./", "http://a/b/c/"),
        ("..", "http://a/b/"),
        ("../", "http://a/b/"),
        ("../g", "http://a/b/g"),
        ("../..", "http://a/"),
        ("../../", "http://a/"),
        ("../../g", "http://a/g"),
        ("../../../g", "http://a/g"),
        ("/./g", "http://a/g"),
        ("/../g", "http://a/g"),
        ("g.", "http://a/b/c/g."),
        ("..g", "http://a/b/c/..g"),
        ("./../g", "http://a/b/g"),
        ("g/./h", "http://a/b/c/g/h"),
        ("g/../h", "http://a/b/c/h"),
        (
            "https://other.example:8443/x.png",
            "https://other.example:8443/x.png",
        ),
        ("HTTP://OTHER.example/x.png", "http://other.example/x.png"),
    ];
    for (location, expected) in table {
        let joined = base.join(location).expect(location);
        assert_eq!(joined.request_target(), *expected, "{location}");
    }
    // A redirect target goes through every rule again.
    assert_eq!(
        base.join("ftp://x/").expect_err("ftp"),
        RemoteUrlRefusal::Scheme {
            scheme: "ftp".to_string()
        }
    );
    assert_eq!(
        base.join("http://u:p@x/").expect_err("credentials"),
        RemoteUrlRefusal::Userinfo
    );
    assert!(matches!(
        base.join("  ").expect_err("empty"),
        RemoteUrlRefusal::Malformed { .. }
    ));
    let to_metadata = base
        .join("http://169.254.169.254/latest/meta-data/")
        .expect("parses");
    assert!(matches!(
        vet_target(&to_metadata, &HostAllowlist::default()),
        Err(RemoteUrlRefusal::NotPublic {
            class: Some(AddressClass::LinkLocal),
            ..
        })
    ));
    // The port of the URL it came from is kept for a relative reference.
    let with_port = parse_remote_image_url("http://h:8080/a/b.png").expect("base");
    assert_eq!(
        with_port.join("c.png").expect("joined").request_target(),
        "http://h:8080/a/c.png"
    );
    assert_eq!(
        with_port.join("//other/").expect("joined").request_target(),
        "http://other/"
    );
}

// ── The allowlist ────────────────────────────────────────────────────────

fn allowlist(entries: &[&str]) -> HostAllowlist {
    HostAllowlist::new(
        entries
            .iter()
            .map(|entry| parse_allowed_host(entry).expect(entry))
            .collect(),
    )
}

fn url(text: &str) -> RemoteImageUrl {
    parse_remote_image_url(text).expect(text)
}

#[test]
fn an_allowlist_entry_matches_its_host_exactly_and_its_port_when_given() {
    let list = allowlist(&[
        "images.intranet",
        "127.0.0.1:8080",
        "[::1]:9000",
        "LocalHost",
    ]);
    for permitted in [
        "http://images.intranet/x.png",
        "https://IMAGES.intranet./x.png",
        "http://images.intranet:1234/x.png",
        "http://127.0.0.1:8080/x.png",
        "http://0x7f.0.0.1:8080/x.png",
        "http://2130706433:8080/x.png",
        "http://[::1]:9000/x.png",
        "http://localhost/x.png",
        "http://LOCALHOST:5/x.png",
    ] {
        assert!(list.permits(&url(permitted)), "{permitted}");
        assert_eq!(
            vet_target(&url(permitted), &list).expect(permitted),
            TargetVerdict::Allowlisted,
            "{permitted}"
        );
    }
    for refused in [
        "http://sub.images.intranet/x.png",
        "http://images.intranet.evil/x.png",
        "http://127.0.0.1/x.png",
        "http://127.0.0.1:8081/x.png",
        "http://[::1]:9001/x.png",
        "http://[::1]/x.png",
        "http://sub.localhost/x.png",
    ] {
        assert!(!list.permits(&url(refused)), "{refused}");
    }
    assert_eq!(
        list.to_string(),
        "images.intranet, 127.0.0.1:8080, [::1]:9000, localhost"
    );
    assert_eq!(HostAllowlist::default().to_string(), "(none)");
    // Duplicates collapse.
    assert_eq!(allowlist(&["a.example", "A.example."]).entries().len(), 1);
}

#[test]
fn a_malformed_allowlist_entry_is_refused_saying_why() {
    for entry in [
        "",
        "   ",
        "http://images.intranet",
        "images.intranet/path",
        "user@images.intranet",
        "images.intranet:0",
        "images.intranet:65536",
        "images.intranet:port",
        "[fe80::1%en0]",
        "[::1",
        "a b",
        "host:1:2",
        "bücher.example",
        "256.1.1.1",
    ] {
        let reason = parse_allowed_host(entry).expect_err(entry);
        assert!(!reason.is_empty(), "{entry}");
    }
    // An unbracketed IPv6 address is an address without a port.
    let entry = parse_allowed_host("::1").expect("v6");
    assert_eq!(entry.host(), &RemoteHost::Ipv6(Ipv6Addr::LOCALHOST));
    assert_eq!(entry.port(), None);
}

// ── Vetting ──────────────────────────────────────────────────────────────

#[test]
fn localhost_names_are_refused_by_name_before_any_resolution() {
    let none = HostAllowlist::default();
    for name in [
        "http://localhost/x.png",
        "http://LOCALHOST./x.png",
        "http://images.localhost/x.png",
        "http://a.b.localhost:8080/x.png",
    ] {
        assert!(
            matches!(
                vet_target(&url(name), &none),
                Err(RemoteUrlRefusal::LocalhostName { .. })
            ),
            "{name}"
        );
    }
    // A name that merely contains the word is an ordinary name.
    assert_eq!(
        vet_target(&url("http://localhost.example.com/x.png"), &none).expect("a name"),
        TargetVerdict::Resolve
    );
    assert_eq!(
        vet_target(&url("http://notlocalhost/x.png"), &none).expect("a name"),
        TargetVerdict::Resolve
    );
}

#[test]
fn every_resolved_address_must_be_public() {
    let host = RemoteHost::Domain("images.example".to_string());
    vet_resolved(&host, &[v4("8.8.8.8"), v6("2606:4700:4700::1111")]).expect("all public");
    for mixed in [
        vec![v4("8.8.8.8"), v4("127.0.0.1")],
        vec![v4("10.0.0.7"), v4("8.8.8.8")],
        vec![v6("2606:4700:4700::1111"), v6("::ffff:10.0.0.1")],
        vec![v4("169.254.169.254")],
    ] {
        let refusal = vet_resolved(&host, &mixed).expect_err("one refused address refuses all");
        assert_eq!(
            refusal,
            RemoteUrlRefusal::NotPublic {
                host: "images.example".to_string(),
                class: None
            }
        );
        let message = refusal.to_string();
        for address in &mixed {
            assert!(
                !message.contains(&address.to_string()),
                "the resolved address is never disclosed: {message}"
            );
        }
        assert!(message.contains("images.example"), "{message}");
        assert!(message.contains("not public"), "{message}");
        assert!(message.contains("--image-url-allow-host"), "{message}");
    }
    // A caller-supplied classification is honoured (a fetcher's tests use
    // one to stand an address in for the public internet).
    vet_resolved_with(&host, &[v4("127.0.0.1")], |_| None).expect("classified public");
}

// ── The three states and the seam ────────────────────────────────────────

/// Records what it was asked for and answers with `body`.
#[derive(Default)]
struct Recording {
    calls: AtomicUsize,
    last_cap: AtomicUsize,
    body: Vec<u8>,
}

impl RemoteImageFetcher for Recording {
    fn fetch(&self, url: &str, max_bytes: usize) -> Result<Vec<u8>, ImageInputError> {
        assert!(url.starts_with("https://"), "{url}");
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.last_cap.store(max_bytes, Ordering::SeqCst);
        Ok(self.body.clone())
    }

    fn describe(&self) -> String {
        "recording".to_string()
    }
}

#[test]
fn the_three_states_stay_distinguishable() {
    let url = "https://example.com/cat.png";
    let disabled = fetch_remote_reference(url, &RemoteImageAccess::Disabled, 100).expect_err("off");
    let without =
        fetch_remote_reference(url, &RemoteImageAccess::OptedInWithoutFetcher, 100).expect_err("");
    assert_eq!(disabled.code(), "image_url_fetch_disabled");
    assert_eq!(without.code(), "image_url_fetch_disabled");
    assert_ne!(
        disabled.to_string(),
        without.to_string(),
        "the reasons differ"
    );
    assert!(!RemoteImageAccess::Disabled.is_opted_in());
    assert!(RemoteImageAccess::OptedInWithoutFetcher.is_opted_in());

    let recording = std::sync::Arc::new(Recording {
        body: vec![7; 10],
        ..Recording::default()
    });
    let shared = SharedRemoteImageFetcher::new(recording.clone());
    let access = RemoteImageAccess::Fetcher(shared.clone());
    assert!(access.is_opted_in());
    assert_eq!(
        fetch_remote_reference(url, &access, 100).expect("fetched"),
        vec![7; 10]
    );
    assert_eq!(recording.calls.load(Ordering::SeqCst), 1);
    assert_eq!(recording.last_cap.load(Ordering::SeqCst), 100);
    // A fetcher that returns more than its cap is held to it anyway.
    let err = fetch_remote_reference(url, &access, 9).expect_err("over the cap");
    assert_eq!(err.code(), "image_too_large");

    // Identity equality, and a Debug line that describes the fetcher.
    assert_eq!(access, RemoteImageAccess::Fetcher(shared.clone()));
    let other = SharedRemoteImageFetcher::new(std::sync::Arc::new(Recording::default()));
    assert_ne!(access, RemoteImageAccess::Fetcher(other));
    assert!(format!("{access:?}").contains("recording"));
    assert_eq!(access.fetcher(), Some(&shared));
    assert_eq!(RemoteImageAccess::Disabled.fetcher(), None);
}

#[test]
fn the_default_preflight_checks_the_syntax_only() {
    let fetcher = Recording::default();
    fetcher
        .preflight("https://example.com/x.png")
        .expect("a well-formed URL");
    let err = fetcher
        .preflight("https://user:pw@example.com/x.png")
        .expect_err("credentials");
    assert_eq!(err.code(), "image_url_refused");
    assert_eq!(
        fetcher.calls.load(Ordering::SeqCst),
        0,
        "nothing is fetched"
    );
}

#[test]
fn refusals_and_failures_carry_their_codes_and_the_shortened_url() {
    let long = format!("https://example.com/{}", "x".repeat(200));
    let refused = RemoteUrlRefusal::Downgrade.into_error(&long);
    assert_eq!(refused.code(), "image_url_refused");
    let failed = RemoteFetchFailure::TimedOut { ms: 250 }.into_error(&long);
    assert_eq!(failed.code(), "image_url_fetch_failed");
    let text = failed.to_string();
    assert!(text.contains("timed out after 250 ms"), "{text}");
    assert!(text.contains(&shown_url(&long)), "{text}");
    assert!(!text.contains(&long), "the URL is shortened: {text}");
    assert_eq!(shown_url(&long).chars().count(), SHOWN_URL_CHARS);
    // Credentials are masked in the echo, whatever the scheme.
    assert_eq!(
        shown_url("http://user:hunter2@example.com/x.png?q=1"),
        "http://***@example.com/x.png?q=1"
    );
    assert_eq!(shown_url("ftp://a@b@host/x"), "ftp://***@host/x");
    assert_eq!(
        shown_url("https://example.com/a@b.png"),
        "https://example.com/a@b.png",
        "an @ in the path is not userinfo"
    );
    let refused = RemoteUrlRefusal::Userinfo.into_error("http://user:hunter2@example.com/x.png");
    assert!(!refused.to_string().contains("hunter2"), "{refused}");
    for (failure, words) in [
        (RemoteFetchFailure::Resolve, "resolve"),
        (RemoteFetchFailure::Connect, "connect"),
        (RemoteFetchFailure::ConnectOrTls, "TLS"),
        (RemoteFetchFailure::Status(404), "status 404"),
        (RemoteFetchFailure::Body, "body"),
        (RemoteFetchFailure::Abandoned, "abandoned"),
    ] {
        assert!(failure.to_string().contains(words), "{failure}");
    }
}

#[test]
fn dot_segments_are_removed_like_rfc_3986_says() {
    for (input, expected) in [
        ("/a/b/c/./../../g", "/a/g"),
        ("/mid/content=5/../6", "/mid/6"),
        ("/a/b/", "/a/b/"),
        ("/.", "/"),
        ("/..", "/"),
        ("/a/..", "/"),
        ("/a/b/..", "/a/"),
        ("/", "/"),
    ] {
        assert_eq!(remove_dot_segments(input), expected, "{input}");
    }
}
