//! RAG-M6: `EvalReport::to_json` swallowed `serde_json` errors into a bare
//! `"{}"` placeholder, and `cmd_eval` wrote that to disk as if it were a
//! real report. `try_to_json` propagates the error instead; `to_json` is
//! kept only as a legacy, non-propagating wrapper so existing callers this
//! package does not own (e.g. `src/cli/cmd_eval.rs`, `src/tests.rs`) keep
//! compiling unchanged.

use oxibonsai_eval::{EvalReport, EvalResultEntry};

fn sample_report() -> EvalReport {
    let mut report = EvalReport::new("test-model");
    report.add(EvalResultEntry {
        task: "wikitext-2".to_string(),
        metric: "perplexity".to_string(),
        value: 12.5,
        unit: "PPL".to_string(),
        notes: None,
    });
    report
}

#[test]
fn try_to_json_succeeds_for_a_normal_report() {
    let report = sample_report();
    let json = report
        .try_to_json()
        .expect("a normal report must serialise");
    assert!(json.contains("test-model"));
    let _: serde_json::Value = serde_json::from_str(&json).expect("must be valid JSON");
}

#[test]
fn to_json_legacy_wrapper_matches_try_to_json_on_success() {
    let report = sample_report();
    assert_eq!(report.to_json(), report.try_to_json().expect("ok"));
}

#[test]
fn try_to_json_and_to_json_agree_on_a_nan_metric_value() {
    // JSON numbers cannot represent NaN or Infinity (RFC 8259). It would be
    // reasonable for `serde_json` to reject them outright, but empirically
    // (serde_json 1.0.150, checked here rather than assumed) it instead
    // serialises a NaN/infinite `f32`/`f64` as the JSON literal `null` —
    // *not* an error. So this struct's specific field types genuinely have
    // no reachable serialisation failure today, and `try_to_json` returning
    // `Ok` here is correct, not a missed case.
    //
    // The contract this test actually locks in: whatever `try_to_json`
    // returns, `to_json` (its non-propagating legacy wrapper) must return
    // the exact same successful string, never silently substituting its
    // `"{}"` placeholder when there was no error to swallow in the first
    // place. `report_extended_tests.rs`'s other tests already exercise the
    // ordinary success path; this one is specifically the "surprising
    // numeric edge case" a caller might expect to fail and doesn't.
    let mut report = EvalReport::new("nan-model");
    report.add(EvalResultEntry {
        task: "broken".to_string(),
        metric: "perplexity".to_string(),
        value: f32::NAN,
        unit: "PPL".to_string(),
        notes: None,
    });

    let json = report
        .try_to_json()
        .expect("serde_json degrades NaN to `null` rather than erroring");
    assert!(
        json.contains("\"value\": null"),
        "NaN should serialise as JSON null: {json}"
    );
    assert_eq!(
        report.to_json(),
        json,
        "to_json must not substitute its \"{{}}\" placeholder when try_to_json succeeded"
    );
}

#[test]
fn to_markdown_and_summary_do_not_panic_on_an_empty_report() {
    let report = EvalReport::new("empty-model");
    let _ = report.to_markdown();
    let _ = report.summary();
    let _ = report
        .try_to_json()
        .expect("an empty report must still serialise");
}
