#![cfg(feature = "server")]
//! T-10: property-based coverage for the stop-sequence matcher
//! (`api_extensions::StopChecker`).
//!
//! ## Split out of `sampler_stop_sequence_proptests.rs` (verifier fix, wave 3)
//!
//! This file used to be part of `sampler_stop_sequence_proptests.rs`, which
//! imported `oxibonsai_runtime::api_extensions::StopChecker` unconditionally
//! even though `api_extensions` is `#[cfg(feature = "server")]`
//! (`crates/oxibonsai-runtime/src/lib.rs`). That made
//! `cargo check -p oxibonsai-runtime --no-default-features --tests` fail
//! with an unresolved-import error (`server` is a default feature, but not
//! every consumer enables defaults) — a build regression against the
//! wave-2.5 addendum's explicit "do not revert, do re-verify" instruction on
//! keeping `--no-default-features` buildable.
//!
//! `StopChecker` genuinely only exists behind the `server` feature, so
//! *this* file (and only this file) needs the guard; the sampler-only
//! properties that never touch `StopChecker` stayed in the original file,
//! ungated, so the T-10 sampler-property deliverable keeps collecting under
//! `--no-default-features` builds too.

use oxibonsai_runtime::api_extensions::StopChecker;
use proptest::prelude::*;

// ── Stop-sequence matcher (`StopChecker`) ───────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// With no configured stop sequences, every text passes through
    /// unmodified and never reports a hit.
    #[test]
    fn stop_checker_with_no_sequences_never_truncates(text in ".*") {
        let checker = StopChecker::new(vec![]);
        prop_assert!(checker.is_empty());
        let (truncated, hit) = checker.truncate_at_stop(&text);
        prop_assert!(!hit);
        prop_assert_eq!(truncated, text);
    }

    /// If `text` genuinely does not contain any configured sequence, it
    /// must pass through unmodified.
    #[test]
    fn stop_checker_passes_through_text_without_any_sequence(
        text in "[a-z]{0,32}",
        sequences in prop::collection::vec("[A-Z]{1,4}", 0..4),
    ) {
        // The generators are disjoint alphabets (lowercase text vs.
        // uppercase sequences), so no sequence can ever occur in `text`.
        let checker = StopChecker::new(sequences);
        let (truncated, hit) = checker.truncate_at_stop(&text);
        prop_assert!(!hit);
        prop_assert_eq!(truncated, text);
    }

    /// `truncate_at_stop`'s two return values are always internally
    /// consistent with `check`, and the truncated text is always a genuine
    /// prefix of the input (never longer, never a different string with
    /// the same length) — the core "safe to truncate" contract this type
    /// exists to provide, exercised over arbitrary text and stop sequences.
    #[test]
    fn stop_checker_truncated_text_is_always_a_prefix(
        text in ".{0,64}",
        sequences in prop::collection::vec(".{1,6}", 0..4),
    ) {
        let checker = StopChecker::new(sequences.clone());
        let (truncated, hit) = checker.truncate_at_stop(&text);

        prop_assert!(
            text.starts_with(&truncated),
            "truncated text {truncated:?} must be a prefix of the original {text:?}"
        );
        prop_assert!(truncated.len() <= text.len());
        prop_assert_eq!(hit, checker.check(&text).is_some(), "truncate_at_stop's hit flag must agree with check()");

        if hit {
            // The remainder must start with *some* configured sequence
            // (the earliest match by byte offset), never a false hit.
            let remainder = &text[truncated.len()..];
            prop_assert!(
                sequences.iter().any(|s| remainder.starts_with(s.as_str())),
                "text after the truncation point ({remainder:?}) must start with a configured sequence"
            );
        } else {
            prop_assert_eq!(&truncated, &text);
        }
    }

    /// A sequence guaranteed to be present (spliced into the middle of the
    /// text) is always found, and the returned prefix never contains it.
    #[test]
    fn stop_checker_finds_a_sequence_guaranteed_present(
        prefix in "[a-z]{0,16}",
        suffix in "[a-z]{0,16}",
        sequence in "[A-Z]{1,6}",
    ) {
        let text = format!("{prefix}{sequence}{suffix}");
        let checker = StopChecker::new(vec![sequence.clone()]);
        let (truncated, hit) = checker.truncate_at_stop(&text);
        prop_assert!(hit, "a spliced-in sequence must always be found");
        prop_assert!(
            !truncated.contains(&sequence),
            "the truncated prefix must not itself contain the matched sequence"
        );
        prop_assert!(text.starts_with(&truncated));
    }
}
