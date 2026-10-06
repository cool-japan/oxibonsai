//! The HTTP client's resolver hook: resolve, vet, and hand the client only
//! what was vetted.
//!
//! `oxihttp`'s client asks a [`DnsResolver`] for the addresses of a host name
//! and dials exactly what it gets back (an IP-literal host never reaches the
//! hook; the parent module vets those). [`VettingResolver`] is built for one
//! request hop: it answers for the one host that hop vetted, resolves it
//! through an [`AddressLookup`], refuses the whole hop when any address is
//! not public (unless the operator allowlisted the host), and returns the
//! vetted addresses. There is no second resolution anywhere for a rebinding
//! name server to answer differently: the addresses checked are the
//! addresses connected to. What happened is kept in a [`ResolveRecord`],
//! which the caller reads to tell a refusal and a resolution failure from a
//! connect failure (the client reports all three as one connect error).

use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use oxibonsai_model::vision::remote::{
    vet_resolved_with, AddressClass, RemoteHost, RemoteUrlRefusal,
};
use oxihttp::OxiHttpError;
use oxihttp_client::resolver::DnsResolver;

/// A boxed address lookup.
pub(crate) type LookupFuture = Pin<Box<dyn Future<Output = std::io::Result<Vec<IpAddr>>> + Send>>;

/// Resolves a host name to its addresses.
pub(crate) trait AddressLookup: Send + Sync + 'static {
    /// The addresses `host` resolves to.
    fn lookup(&self, host: &str) -> LookupFuture;
}

/// The operating system's resolver (`getaddrinfo`, on tokio's blocking
/// pool of the fetcher's own runtime).
pub(super) struct SystemLookup;

impl AddressLookup for SystemLookup {
    fn lookup(&self, host: &str) -> LookupFuture {
        let host = host.to_string();
        Box::pin(async move {
            let addresses = tokio::net::lookup_host((host.as_str(), 0u16)).await?;
            Ok(addresses.map(|address| address.ip()).collect())
        })
    }
}

/// What the resolver hook did for one hop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ResolveOutcome {
    /// Resolved, and every address passed (or the host is allowlisted).
    Resolved(Vec<IpAddr>),
    /// Refused by the policy; the client was handed nothing.
    Refused(RemoteUrlRefusal),
    /// The name did not resolve, or resolved to nothing.
    Failed,
}

/// The resolver hook's record for one hop (see the module docs).
#[derive(Debug, Default)]
pub(super) struct ResolveRecord {
    outcome: Mutex<Option<ResolveOutcome>>,
}

impl ResolveRecord {
    /// What happened, if the hook ran.
    pub(super) fn outcome(&self) -> Option<ResolveOutcome> {
        self.outcome.lock().ok().and_then(|outcome| outcome.clone())
    }

    fn set(&self, outcome: ResolveOutcome) {
        if let Ok(mut slot) = self.outcome.lock() {
            *slot = Some(outcome);
        }
    }
}

/// The resolver hook of one hop's client (see the module docs).
pub(super) struct VettingResolver {
    /// The host the hop vetted; the only name this resolver answers for.
    pub(super) host: RemoteHost,
    /// The operator allowlisted the host: no address class applies.
    pub(super) allowlisted: bool,
    /// How the name is resolved.
    pub(super) lookup: Arc<dyn AddressLookup>,
    /// How each address is classified.
    pub(super) classify: fn(IpAddr) -> Option<AddressClass>,
    /// What happened.
    pub(super) record: Arc<ResolveRecord>,
}

impl DnsResolver for VettingResolver {
    fn resolve(
        &self,
        name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<SocketAddr>, OxiHttpError>> + Send>> {
        let record = Arc::clone(&self.record);
        if !name.eq_ignore_ascii_case(&self.host.resolver_name()) {
            // The client only ever asks for the host it was built for; any
            // other name is refused rather than resolved unvetted.
            record.set(ResolveOutcome::Refused(RemoteUrlRefusal::UnexpectedHost));
            return Box::pin(async {
                Err(OxiHttpError::Dns(
                    "refused: a name other than the vetted host".to_string(),
                ))
            });
        }
        let host = self.host.clone();
        let allowlisted = self.allowlisted;
        let lookup = Arc::clone(&self.lookup);
        let classify = self.classify;
        let name = name.to_string();
        Box::pin(async move {
            let addresses = match lookup.lookup(&name).await {
                Ok(addresses) if !addresses.is_empty() => addresses,
                Ok(_) | Err(_) => {
                    record.set(ResolveOutcome::Failed);
                    return Err(OxiHttpError::Dns("the name did not resolve".to_string()));
                }
            };
            if !allowlisted {
                if let Err(refusal) = vet_resolved_with(&host, &addresses, classify) {
                    record.set(ResolveOutcome::Refused(refusal));
                    return Err(OxiHttpError::Dns(
                        "refused: an address that is not public".to_string(),
                    ));
                }
            }
            record.set(ResolveOutcome::Resolved(addresses.clone()));
            // Port 0: the client sets the URL's port on every address.
            Ok(addresses
                .into_iter()
                .map(|address| SocketAddr::new(address, 0))
                .collect())
        })
    }
}
