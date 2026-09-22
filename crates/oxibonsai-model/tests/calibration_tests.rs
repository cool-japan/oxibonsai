#![cfg(feature = "training")]
//! Integration tests for the PTQ calibration pipeline.

use oxibonsai_model::calibration::{
    simulate_calibration, validate_calibration, CalibMethod, CalibValidation, CalibrationDb,
    LayerCalibStats,
};

// ─── LayerCalibStats tests ────────────────────────────────────────────────────

#[test]
fn layer_calib_stats_new() {
    let stats = LayerCalibStats::new("block0.attn.q");
    assert_eq!(stats.layer_name, "block0.attn.q");
    assert_eq!(stats.num_samples, 0);
    assert_eq!(stats.running_mean, 0.0);
    assert_eq!(stats.running_var, 0.0);
}

#[test]
fn layer_calib_stats_update_single() {
    let mut stats = LayerCalibStats::new("layer0");
    let data: Vec<f32> = (0..128).map(|i| i as f32 * 0.01 - 0.64).collect();
    let n = data.len();
    stats.update(&data);
    assert_eq!(
        stats.num_samples, n,
        "num_samples should equal data.len() after one update"
    );
}

#[test]
fn layer_calib_stats_running_min_max() {
    let mut stats = LayerCalibStats::new("layer0");
    stats.update(&[-3.0_f32, 1.0, 2.5]);
    stats.update(&[0.0, 5.0, -1.0]);
    assert!(
        (stats.running_min - (-3.0)).abs() < 1e-6,
        "running_min should be -3.0, got {}",
        stats.running_min
    );
    assert!(
        (stats.running_max - 5.0).abs() < 1e-6,
        "running_max should be 5.0, got {}",
        stats.running_max
    );
}

#[test]
fn layer_calib_stats_std_dev() {
    let mut stats = LayerCalibStats::new("layer0");
    let data: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.01).collect();
    stats.update(&data);
    let sd = stats.std_dev();
    assert!(
        sd > 0.0,
        "std_dev should be positive for non-constant data, got {sd}"
    );
}

#[test]
fn layer_calib_stats_percentile_abs() {
    let mut stats = LayerCalibStats::new("layer0");
    let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    stats.update(&data);
    // p=1.0 should give near max(|x|) = 100
    let p100 = stats.percentile_abs(1.0);
    assert!(
        p100 >= 99.0,
        "p=1.0 percentile should be near 100.0, got {p100}"
    );
    // p=0.5 should be well below 100
    let p50 = stats.percentile_abs(0.5);
    assert!(p50 < p100, "p50={p50} should be less than p100={p100}");
}

#[test]
fn layer_calib_stats_aciq_clip() {
    let mut stats = LayerCalibStats::new("layer0");
    let data: Vec<f32> = (0..500).map(|i| (i as f32 - 250.0) * 0.1).collect();
    stats.update(&data);
    let sd = stats.std_dev();
    let aciq = stats.aciq_clip();
    let expected = 2.83 * sd;
    assert!(
        (aciq - expected).abs() < 1e-4,
        "aciq_clip={aciq}, expected 2.83*std_dev={expected}"
    );
}

#[test]
fn layer_calib_stats_compute_scale_minmax() {
    let mut stats = LayerCalibStats::new("layer0");
    stats.update(&[-2.54_f32, 1.0, 0.5, -0.3, std::f32::consts::PI]);
    let scale = stats.compute_scale(CalibMethod::MinMax);
    assert!(scale > 0.0, "MinMax scale must be positive, got {scale}");
    // scale should be abs_max / 127 = PI / 127
    let expected = std::f32::consts::PI / 127.0;
    assert!(
        (scale - expected).abs() < 1e-5,
        "MinMax scale={scale}, expected ~{expected}"
    );
}

#[test]
fn layer_calib_stats_compute_scale_percentile() {
    let mut stats = LayerCalibStats::new("layer0");
    // Data with one large outlier at the end
    let mut data: Vec<f32> = (0..999).map(|i| (i as f32) * 0.001).collect();
    data.push(100.0); // single extreme outlier
    stats.update(&data);
    let scale_minmax = stats.compute_scale(CalibMethod::MinMax);
    let scale_p99 = stats.compute_scale(CalibMethod::Percentile(0.99));
    assert!(
        scale_p99 <= scale_minmax,
        "Percentile(0.99) scale {scale_p99} should be <= MinMax scale {scale_minmax}"
    );
}

// ─── CalibSummary tests ───────────────────────────────────────────────────────

#[test]
fn calib_summary_summary_line_nonempty() {
    let mut stats = LayerCalibStats::new("fc1");
    stats.update(&[0.1_f32, -0.2, 0.3, -0.1, 0.5]);
    let summary = stats.summary();
    let line = summary.summary_line();
    assert!(!line.is_empty(), "summary_line should not be empty");
    assert!(
        line.contains("fc1"),
        "summary_line should contain layer name 'fc1', got: {line}"
    );
    assert!(
        line.contains("n=5"),
        "summary_line should contain num_samples=5, got: {line}"
    );
}

// ─── CalibrationDb tests ──────────────────────────────────────────────────────

#[test]
fn calib_db_new_minmax() {
    let db = CalibrationDb::new_minmax();
    assert_eq!(db.num_layers(), 0, "fresh db should have 0 layers");
}

#[test]
fn calib_db_record_creates_layer() {
    let mut db = CalibrationDb::new_minmax();
    assert_eq!(db.num_layers(), 0);

    db.record("attn.q", &[0.1_f32, -0.2, 0.5]);
    assert_eq!(db.num_layers(), 1, "should have 1 layer after first record");

    db.record("attn.k", &[0.3_f32, 0.4]);
    assert_eq!(
        db.num_layers(),
        2,
        "should have 2 layers after second record"
    );

    // Recording to existing layer should NOT increase count
    db.record("attn.q", &[0.9_f32, -0.1]);
    assert_eq!(
        db.num_layers(),
        2,
        "re-recording same layer should not increase count"
    );

    // Verify samples accumulated
    let stats = db.get_stats("attn.q").expect("attn.q should exist");
    assert_eq!(stats.num_samples, 5, "attn.q should have 3+2=5 samples");
}

#[test]
fn calib_db_scale_for_unknown_layer() {
    let db = CalibrationDb::new_minmax();
    let result = db.scale_for_layer("nonexistent_layer_xyz");
    assert!(
        result.is_none(),
        "Should return None for unknown layer, got {:?}",
        result
    );
}

#[test]
fn calib_db_export_scales_all_layers() {
    let mut db = CalibrationDb::new_minmax();
    db.record("layer_a", &[1.0_f32, 2.0, 3.0]);
    db.record("layer_b", &[-1.0_f32, 0.5]);
    db.record("layer_c", &[0.1_f32, 0.2]);

    let scales = db.export_scales();
    assert_eq!(
        scales.len(),
        db.num_layers(),
        "export_scales count {} should match num_layers {}",
        scales.len(),
        db.num_layers()
    );
    assert!(scales.contains_key("layer_a"), "should contain layer_a");
    assert!(scales.contains_key("layer_b"), "should contain layer_b");
    assert!(scales.contains_key("layer_c"), "should contain layer_c");

    // All scales should be positive
    for (name, &scale) in &scales {
        assert!(
            scale > 0.0,
            "scale for {} should be positive, got {}",
            name,
            scale
        );
    }
}

#[test]
fn calib_db_report_nonempty() {
    let mut db = CalibrationDb::new_percentile(0.999);
    db.record("block0.ffn", &[0.1_f32, -0.5, 0.3, 0.9, -0.8]);
    db.record("block1.attn", &[-0.3_f32, 0.7, 0.2]);

    let report = db.report();
    assert!(
        !report.is_empty(),
        "report() should return non-empty string"
    );
    assert!(
        report.contains("block0.ffn"),
        "report should mention block0.ffn"
    );
    assert!(
        report.contains("block1.attn"),
        "report should mention block1.attn"
    );
    assert!(
        report.contains("Percentile"),
        "report should mention calibration method"
    );
}

// ─── simulate_calibration tests ──────────────────────────────────────────────

#[test]
fn simulate_calibration_fills_db() {
    let mut db = CalibrationDb::new_minmax();
    let layer_names = ["layer0", "layer1", "layer2", "layer3"];
    simulate_calibration(&mut db, &layer_names, 256, 42);

    assert_eq!(
        db.num_layers(),
        layer_names.len(),
        "db should have one entry per layer name"
    );

    for &name in &layer_names {
        let stats = db
            .get_stats(name)
            .expect("layer should exist after simulation");
        assert_eq!(
            stats.num_samples, 256,
            "layer {name} should have 256 samples, got {}",
            stats.num_samples
        );
        assert!(
            stats.running_max > stats.running_min,
            "max should exceed min"
        );
    }
}

#[test]
fn simulate_calibration_deterministic() {
    let layer_names = ["attn.q", "attn.k", "attn.v", "ffn.up", "ffn.down"];
    let seed = 12345u64;

    let mut db1 = CalibrationDb::new_minmax();
    simulate_calibration(&mut db1, &layer_names, 128, seed);
    let scales1 = db1.export_scales();

    let mut db2 = CalibrationDb::new_minmax();
    simulate_calibration(&mut db2, &layer_names, 128, seed);
    let scales2 = db2.export_scales();

    for name in &layer_names {
        let s1 = scales1[*name];
        let s2 = scales2[*name];
        assert!(
            (s1 - s2).abs() < 1e-8,
            "scales should be bit-identical for same seed: layer={name}, s1={s1}, s2={s2}"
        );
    }
}

// ─── CalibValidation tests ────────────────────────────────────────────────────

#[test]
fn validate_calibration_valid_layer() {
    let mut stats = LayerCalibStats::new("block0.attn");
    let data: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.05).collect();
    stats.update(&data);
    let scale = stats.compute_scale(CalibMethod::MinMax);
    let val = CalibValidation::validate("block0.attn", &stats, scale);
    assert!(
        val.is_valid,
        "should be valid for reasonable activation data, issues={:?}",
        val.issues
    );
    assert!(
        val.issues.is_empty(),
        "no issues expected, got: {:?}",
        val.issues
    );
}

#[test]
fn validate_calibration_all_valid() {
    let mut db = CalibrationDb::new_minmax();
    let layer_names = [
        "embed",
        "block0.attn.q",
        "block0.attn.k",
        "block0.ffn.up",
        "block0.ffn.down",
        "lm_head",
    ];
    simulate_calibration(&mut db, &layer_names, 512, 999);

    let validations = validate_calibration(&db);
    assert_eq!(
        validations.len(),
        layer_names.len(),
        "should have one validation per layer"
    );

    for v in &validations {
        assert!(
            v.is_valid,
            "layer '{}' should be valid after simulate_calibration, issues={:?}",
            v.layer_name, v.issues
        );
        assert!(
            v.scale > 0.0,
            "scale should be positive for layer '{}'",
            v.layer_name
        );
    }
}
