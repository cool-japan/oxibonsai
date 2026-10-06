//! One-time logging of the kernel-tier selection.
//!
//! [`KernelDispatcher::auto_detect`] runs once for every engine that picks its
//! own dispatcher, and a server that keeps a pool of engines (each building
//! more than one dispatcher) constructs many dispatchers that all reach the
//! same outcome. Announcing every one at `INFO` printed the same `selected
//! kernel tier` line eight times at `serve` startup (four replicas, two
//! constructions each).
//!
//! `KernelDispatcher::auto_detect` keeps a process-wide registry of the
//! `(tier, reason)` pairs already announced. The first construction that
//! selects a pair logs at `INFO`; every later construction that selects the
//! same pair logs at `DEBUG` with the same fields, so the detail is still
//! there when someone asks for it. A different tier, or the same tier for a
//! different reason (auto-detected versus explicitly requested CPU, say), is
//! a different pair and announces itself at `INFO` again.
//!
//! Both events carry the message `selected kernel tier` and two fields:
//! `tier` (the tier's display name) and `reason`
//! ([`KernelDispatcher::effective_tier_reason`]).

use std::sync::{Mutex, PoisonError};

use crate::dispatch::KernelDispatcher;
use crate::tier::KernelTier;

/// The `(tier, reason)` pairs already announced at `INFO`.
///
/// [`KernelTier`] is not `Hash`, and only a handful of pairs can ever exist
/// (a few tiers times a few reasons), so a `Vec` scanned linearly is the whole
/// registry.
struct TierLogRegistry {
    announced: Mutex<Vec<(KernelTier, String)>>,
}

impl TierLogRegistry {
    const fn new() -> Self {
        Self {
            announced: Mutex::new(Vec::new()),
        }
    }

    /// Record `(tier, reason)` and report whether this is the first time it
    /// has been seen. Atomic: of any number of concurrent callers passing the
    /// same pair, exactly one gets `true`.
    fn first_time(&self, tier: KernelTier, reason: &str) -> bool {
        let mut announced = self
            .announced
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if announced
            .iter()
            .any(|(seen_tier, seen_reason)| *seen_tier == tier && seen_reason == reason)
        {
            return false;
        }
        announced.push((tier, reason.to_string()));
        true
    }
}

/// The process-wide registry.
static GLOBAL_REGISTRY: TierLogRegistry = TierLogRegistry::new();

#[cfg(test)]
thread_local! {
    /// A registry installed for the current test thread only (see
    /// [`tests::with_fresh_registry`]): the process-wide registry is shared by
    /// every test that builds a dispatcher on any thread, so a test that
    /// counts events needs a registry of its own to be deterministic.
    static THREAD_REGISTRY: std::cell::RefCell<Option<std::sync::Arc<TierLogRegistry>>> =
        const { std::cell::RefCell::new(None) };
}

/// Emit the selection event for `(tier, reason)` against `registry`: `INFO`
/// the first time the pair is seen, `DEBUG` afterwards.
fn log_in(registry: &TierLogRegistry, tier: KernelTier, reason: &str) {
    if registry.first_time(tier, reason) {
        tracing::info!(tier = %tier, reason = %reason, "selected kernel tier");
    } else {
        tracing::debug!(tier = %tier, reason = %reason, "selected kernel tier");
    }
}

/// Announce that `tier` was selected for `reason`: `INFO` once per process
/// per `(tier, reason)` pair, `DEBUG` for every repeat (module docs).
pub(crate) fn log_tier_selected(tier: KernelTier, reason: &str) {
    #[cfg(test)]
    {
        let scoped = THREAD_REGISTRY.with(|slot| slot.borrow().clone());
        if let Some(registry) = scoped {
            log_in(&registry, tier, reason);
            return;
        }
    }
    log_in(&GLOBAL_REGISTRY, tier, reason);
}

impl KernelDispatcher {
    /// Announce this dispatcher's tier selection (see [`log_tier_selected`]).
    /// Called by [`KernelDispatcher::auto_detect`], the constructor that
    /// chooses a tier for the caller.
    pub(crate) fn log_selection(&self) {
        log_tier_selected(self.tier(), &self.effective_tier_reason());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing::{Level, Subscriber};

    use super::*;

    /// Run `f` with a registry that only this thread sees, so the events it
    /// counts cannot be pre-empted by another test's dispatcher construction.
    fn with_fresh_registry<T>(f: impl FnOnce() -> T) -> T {
        THREAD_REGISTRY.with(|slot| *slot.borrow_mut() = Some(Arc::new(TierLogRegistry::new())));
        let result = f();
        THREAD_REGISTRY.with(|slot| *slot.borrow_mut() = None);
        result
    }

    /// One captured `selected kernel tier` event.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Captured {
        level: Level,
        message: String,
        tier: Option<String>,
        reason: Option<String>,
    }

    /// Collects the fields of one event.
    #[derive(Default)]
    struct FieldCollector {
        message: Option<String>,
        tier: Option<String>,
        reason: Option<String>,
    }

    impl Visit for FieldCollector {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            let rendered = format!("{value:?}");
            match field.name() {
                "message" => self.message = Some(rendered),
                "tier" => self.tier = Some(rendered),
                "reason" => self.reason = Some(rendered),
                _ => {}
            }
        }
    }

    /// A subscriber that records every event down to `DEBUG` on the thread
    /// it is installed for.
    struct Capture {
        events: Arc<Mutex<Vec<Captured>>>,
    }

    impl Subscriber for Capture {
        fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
            *metadata.level() <= Level::DEBUG
        }
        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            let mut fields = FieldCollector::default();
            event.record(&mut fields);
            self.events
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(Captured {
                    level: *event.metadata().level(),
                    message: fields.message.unwrap_or_default(),
                    tier: fields.tier,
                    reason: fields.reason,
                });
        }
        fn enter(&self, _span: &tracing::span::Id) {}
        fn exit(&self, _span: &tracing::span::Id) {}
    }

    /// Run `f` under a fresh registry and a capturing subscriber and return
    /// the `selected kernel tier` events it produced, in order (other events,
    /// such as the one-time "GPU unavailable" warning, are not selection
    /// events and are left out).
    fn capture_selection_events(f: impl FnOnce()) -> Vec<Captured> {
        let events = Arc::new(Mutex::new(Vec::new()));
        let subscriber = Capture {
            events: Arc::clone(&events),
        };
        with_fresh_registry(|| tracing::subscriber::with_default(subscriber, f));
        let captured = events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        captured
            .into_iter()
            .filter(|event| event.message == "selected kernel tier")
            .collect()
    }

    fn levels(events: &[Captured]) -> Vec<Level> {
        events.iter().map(|event| event.level).collect()
    }

    /// A tier other than `Reference` that every supported architecture has.
    #[cfg(target_arch = "aarch64")]
    const OTHER_TIER: KernelTier = KernelTier::Neon;
    #[cfg(target_arch = "x86_64")]
    const OTHER_TIER: KernelTier = KernelTier::Avx2;

    #[test]
    fn a_pair_is_new_exactly_once() {
        let registry = TierLogRegistry::new();
        assert!(registry.first_time(KernelTier::Reference, "requested"));
        assert!(!registry.first_time(KernelTier::Reference, "requested"));
        assert!(!registry.first_time(KernelTier::Reference, "requested"));
        // Same tier, another reason: a new pair.
        assert!(registry.first_time(KernelTier::Reference, "auto-detected"));
        assert!(!registry.first_time(KernelTier::Reference, "auto-detected"));
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn a_different_tier_is_a_new_pair() {
        let registry = TierLogRegistry::new();
        assert!(registry.first_time(KernelTier::Reference, "same reason"));
        assert!(registry.first_time(OTHER_TIER, "same reason"));
        assert!(!registry.first_time(OTHER_TIER, "same reason"));
        assert!(!registry.first_time(KernelTier::Reference, "same reason"));
    }

    #[test]
    fn concurrent_callers_of_one_pair_get_exactly_one_first_time() {
        let registry = TierLogRegistry::new();
        let firsts = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|_| scope.spawn(|| registry.first_time(KernelTier::Reference, "contended")))
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap_or(false))
                .filter(|first| *first)
                .count()
        });
        assert_eq!(firsts, 1, "exactly one of the concurrent callers is first");
        assert!(!registry.first_time(KernelTier::Reference, "contended"));
    }

    #[test]
    fn three_dispatchers_with_the_same_tier_log_one_info_and_two_debug_events() {
        let events = capture_selection_events(|| {
            let first = KernelDispatcher::auto_detect();
            let second = KernelDispatcher::auto_detect();
            let third = KernelDispatcher::auto_detect();
            assert_eq!(first.tier(), second.tier());
            assert_eq!(second.tier(), third.tier());
        });
        assert_eq!(
            levels(&events),
            [Level::INFO, Level::DEBUG, Level::DEBUG],
            "the first construction announces at INFO, the repeats at DEBUG: {events:?}"
        );
        // The DEBUG repeats carry exactly the INFO event's fields.
        assert!(
            events[0].tier.is_some() && events[0].reason.is_some(),
            "the event must carry `tier` and `reason` fields: {:?}",
            events[0]
        );
        for repeat in &events[1..] {
            assert_eq!(repeat.tier, events[0].tier);
            assert_eq!(repeat.reason, events[0].reason);
        }
    }

    #[test]
    fn the_logged_fields_are_the_dispatchers_own_tier_and_reason() {
        let mut expected = None;
        let events = capture_selection_events(|| {
            let dispatcher = KernelDispatcher::auto_detect();
            expected = Some((
                dispatcher.tier().to_string(),
                dispatcher.effective_tier_reason(),
            ));
        });
        let (tier, reason) = expected.unwrap_or_default();
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0].tier.as_deref(), Some(tier.as_str()));
        assert_eq!(events[0].reason.as_deref(), Some(reason.as_str()));
    }

    #[test]
    fn a_different_reason_for_the_same_construction_path_announces_again() {
        // The default `auto_detect` and an `auto_detect` under an explicit
        // CPU request are different `(tier, reason)` pairs (a different tier
        // on a GPU host, a different reason on a CPU-only one), so the second
        // announces at INFO although the process already selected a tier.
        let events = capture_selection_events(|| {
            let _default = KernelDispatcher::auto_detect();
            {
                let _scope = crate::gpu_backend::CpuOnlyBackendScope::enter();
                let _scoped = KernelDispatcher::auto_detect();
                let _scoped_again = KernelDispatcher::auto_detect();
            }
            let _default_again = KernelDispatcher::auto_detect();
        });
        assert_eq!(
            levels(&events),
            [Level::INFO, Level::INFO, Level::DEBUG, Level::DEBUG],
            "{events:?}"
        );
        assert_ne!(
            (&events[0].tier, &events[0].reason),
            (&events[1].tier, &events[1].reason),
            "the explicit-CPU selection must differ from the default one"
        );
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn a_different_tier_produces_a_new_info_line() {
        let events = capture_selection_events(|| {
            log_tier_selected(KernelTier::Reference, "requested");
            log_tier_selected(KernelTier::Reference, "requested");
            log_tier_selected(OTHER_TIER, "requested");
            log_tier_selected(OTHER_TIER, "requested");
        });
        assert_eq!(
            levels(&events),
            [Level::INFO, Level::DEBUG, Level::INFO, Level::DEBUG],
            "{events:?}"
        );
        assert_eq!(events[0].tier.as_deref(), Some("reference"));
        assert_eq!(
            events[2].tier.as_deref(),
            Some(OTHER_TIER.to_string().as_str())
        );
    }

    #[test]
    fn explicit_tier_constructors_do_not_announce() {
        let events = capture_selection_events(|| {
            let _requested = KernelDispatcher::with_tier(KernelTier::Reference);
            let _try = KernelDispatcher::try_with_tier(KernelTier::Reference);
        });
        assert!(
            events.is_empty(),
            "with_tier/try_with_tier pick nothing, so they log nothing: {events:?}"
        );
    }
}
