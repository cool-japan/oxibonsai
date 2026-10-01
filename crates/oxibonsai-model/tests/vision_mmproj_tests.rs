//! Acceptance tests for the Qwen3-VL vision tower (bonsai2-design.md §6,
//! §8.2): the real Bonsai 2 vision projector, the synthetic one through the
//! public API, and the guarantee that text-only inference neither needs nor
//! notices a projector.
//!
//! # The real projector
//!
//! `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (0.63 GB) is found through
//! `$OXI_BONSAI2_MMPROJ_GGUF`, else under the workspace `models/` directory
//! (or `$OXIBONSAI_MODELS_DIR`) via the testkit resolver. When it is absent
//! the real-weight test records a `bonsai2-mmproj` / `executed: false`
//! capability line and returns — unless `OXI_REQUIRE_MODEL_FILES=1`, which
//! turns the absence into a failure. `executed: true` is recorded only after
//! every assertion has passed.
//!
//! The `f64` ground truth is `src/vision/f64_reference.rs`, included here by
//! path: it depends on nothing in this crate, so the same file serves the
//! crate's unit tests and this real-weight check.

#[path = "../src/vision/f64_reference.rs"]
mod f64_reference;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
use oxibonsai_kernels::{KernelDispatcher, KernelTier};
use oxibonsai_model::vision::{GridSize, ImageRgb8, VisionTower};
use oxibonsai_model::BonsaiModel;
use oxibonsai_testkit::capability::{record_executed_timed, record_skipped, Capability};
use oxibonsai_testkit::gguf_fixture::tiny_dense_qwen3_gguf;
use oxibonsai_testkit::mmproj_fixture::{pattern_rgb8, synthetic_mmproj_gguf, MmprojFixtureSpec};

/// Names the real projector file.
const MMPROJ_ENV: &str = "OXI_BONSAI2_MMPROJ_GGUF";
/// Release file name of the real projector.
const MMPROJ_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";
/// `1` turns a missing model file into a failure.
const REQUIRE_ENV: &str = "OXI_REQUIRE_MODEL_FILES";
/// The acceptance bound on the per-row relative L2 error against `f64`.
const F64_BOUND: f64 = 1e-3;
/// Resident-weight ceiling for the real projector.
const RESIDENT_CEILING: usize = 2_500_000_000;

const REAL_TEST: &str =
    "oxibonsai-model::vision_mmproj_tests::real_mmproj_binds_every_block_and_matches_the_f64_reference";

fn locate_mmproj() -> Option<PathBuf> {
    if let Ok(path) = std::env::var(MMPROJ_ENV) {
        if !path.trim().is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    oxibonsai_testkit::workspace::find_model(MMPROJ_FILE)
}

fn require_model_files() -> bool {
    std::env::var(REQUIRE_ENV).is_ok_and(|v| v.trim() == "1")
}

fn image(width: usize, height: usize) -> ImageRgb8 {
    ImageRgb8::new(width, height, pattern_rgb8(width, height)).expect("pattern image")
}

/// Finite, non-degenerate rows: every value finite, row L2 norms within a
/// factor of 10 of each other, and no two distinct rows with cosine
/// similarity of 0.999 or more.
fn assert_non_degenerate(rows: &[f32], dim: usize, label: &str) {
    assert!(
        rows.iter().all(|v| v.is_finite()),
        "{label}: non-finite value"
    );
    let rows: Vec<&[f32]> = rows.chunks(dim).collect();
    let norms: Vec<f64> = rows
        .iter()
        .map(|r| {
            r.iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let max = norms.iter().copied().fold(0.0, f64::max);
    let min = norms.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!(
        "{label}: {} rows, row norms {min:.3} .. {max:.3} (ratio {:.2})",
        rows.len(),
        max / min
    );
    assert!(
        min > 0.0 && max / min <= 10.0,
        "{label}: row norms {min} .. {max}"
    );
    let mut worst = f64::NEG_INFINITY;
    for i in 0..rows.len() {
        for j in i + 1..rows.len() {
            let dot: f64 = rows[i]
                .iter()
                .zip(rows[j])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let cos = dot / (norms[i] * norms[j]);
            worst = worst.max(cos);
            assert!(cos < 0.999, "{label}: rows {i} and {j} have cosine {cos}");
        }
    }
    eprintln!("{label}: largest cosine between distinct rows {worst:.4}");
}

/// Print a `CAPABILITY-REPORT` line straight to the process's standard
/// error. libtest captures `eprintln!` output of a passing test; a direct
/// write is not captured, so the evidence line shows up in every run's log,
/// next to the `test result` line, not only under `--nocapture`.
fn report_capability(line: &str) {
    use std::io::Write as _;
    let mut stderr = std::io::stderr().lock();
    // Bookkeeping only: a failed diagnostic write must not fail the test.
    let _ = writeln!(stderr, "{line}");
}

/// Skip (with a capability record) or, under `OXI_REQUIRE_MODEL_FILES=1`,
/// fail.
fn skip_real(test: &str) {
    assert!(
        !require_model_files(),
        "{test}: {MMPROJ_FILE} not found (set {MMPROJ_ENV}) and {REQUIRE_ENV}=1"
    );
    record_skipped(Capability::Bonsai2Mmproj, test);
    report_capability(&format!(
        "CAPABILITY-REPORT capability={} executed=false test={test} \
         (no {MMPROJ_FILE}; set {MMPROJ_ENV})",
        Capability::Bonsai2Mmproj
    ));
}

/// The real Bonsai 2 projector: every block bound, the resident size, a
/// 64 x 64 and a 256 x 192 image encoded to the expected grids with sane
/// rows, the 64 x 64 (and 256 x 192) result against the `f64` reference of
/// the same graph, and the wall time of a full-resolution 768 x 768 image.
///
/// One test function on purpose: the tower holds about 1.8 GB of `f32`
/// weights, and a second test loading its own copy in parallel would double
/// that.
#[test]
fn real_mmproj_binds_every_block_and_matches_the_f64_reference() {
    let Some(path) = locate_mmproj() else {
        skip_real(REAL_TEST);
        return;
    };
    assert!(
        path.is_file(),
        "{MMPROJ_ENV} names {} but no such file exists",
        path.display()
    );
    let started = Instant::now();
    let mmap = mmap_gguf_file(&path).expect("map the projector");
    let gguf = GgufFile::parse(&mmap).expect("parse the projector");
    eprintln!("projector: {} ({} bytes)", path.display(), mmap.len());

    let load_started = Instant::now();
    let tower = VisionTower::from_mmproj(&gguf).expect("load the projector");
    let load_time = load_started.elapsed();
    let cfg = tower.config();
    assert_eq!(tower.block_count(), 27, "ViT blocks bound");
    assert_eq!(gguf.tensors.len(), 334);
    assert_eq!(tower.bound_tensor_count(), gguf.tensors.len());
    assert_eq!(
        (cfg.hidden, cfg.heads, cfg.head_dim, cfg.ffn, cfg.blocks),
        (1152, 16, 72, 4304, 27)
    );
    assert_eq!(
        (
            cfg.patch_size,
            cfg.spatial_merge,
            cfg.image_size,
            cfg.pos_grid
        ),
        (16, 2, 768, 48)
    );
    assert_eq!((cfg.merger_hidden, cfg.projection_dim), (4608, 5120));
    assert!((cfg.eps - 1e-6).abs() < 1e-9, "eps {}", cfg.eps);
    assert_eq!(cfg.image_mean, [0.5; 3]);
    assert_eq!(cfg.image_std, [0.5; 3]);
    let resident = tower.resident_bytes();
    eprintln!(
        "loaded in {:.2} s; resident f32 weights {resident} bytes ({:.3} GiB)",
        load_time.as_secs_f64(),
        resident as f64 / f64::from(1u32 << 30)
    );
    assert!(resident <= RESIDENT_CEILING, "resident {resident} bytes");

    // 64 x 64: a 2 x 2 merged grid, checked against the f64 reference.
    let small = image(64, 64);
    let (rows, grid) = tower.encode(&small, 1024).expect("encode 64x64");
    assert_eq!(grid, GridSize { h: 2, w: 2 });
    assert_eq!(rows.len(), 4 * 5120);
    assert_non_degenerate(&rows, 5120, "real 64x64");
    let f64_started = Instant::now();
    let reference =
        f64_reference::encode_rgb8(&gguf, 64, 64, &small.data).expect("f64 reference 64x64");
    assert_eq!((reference.grid_h, reference.grid_w), (grid.h, grid.w));
    let err_small = f64_reference::max_row_relative_error(&rows, &reference.rows, reference.dim);
    eprintln!(
        "real 64x64: max per-row relative error vs f64 {err_small:.3e} (reference took {:.2} s)",
        f64_started.elapsed().as_secs_f64()
    );
    assert!(
        err_small <= F64_BOUND,
        "real 64x64: {err_small:.3e} > {F64_BOUND:e}"
    );

    // 256 x 192: a 6 x 8 merged grid (rows x columns), the one non-square
    // real-weight case, also against the f64 reference.
    let wide = image(256, 192);
    let (rows, grid) = tower.encode(&wide, 1024).expect("encode 256x192");
    assert_eq!(grid, GridSize { h: 6, w: 8 });
    assert_eq!(rows.len(), 48 * 5120);
    assert_non_degenerate(&rows, 5120, "real 256x192");
    let f64_started = Instant::now();
    let reference =
        f64_reference::encode_rgb8(&gguf, 256, 192, &wide.data).expect("f64 reference 256x192");
    assert_eq!((reference.grid_h, reference.grid_w), (grid.h, grid.w));
    let err_wide = f64_reference::max_row_relative_error(&rows, &reference.rows, reference.dim);
    eprintln!(
        "real 256x192: max per-row relative error vs f64 {err_wide:.3e} (reference took {:.2} s)",
        f64_started.elapsed().as_secs_f64()
    );
    assert!(
        err_wide <= F64_BOUND,
        "real 256x192: {err_wide:.3e} > {F64_BOUND:e}"
    );

    // The native resolution: 48 x 48 patches, 576 merged tokens.
    let full = image(768, 768);
    let encode_started = Instant::now();
    let (rows, grid) = tower.encode(&full, 1024).expect("encode 768x768");
    let encode_time: Duration = encode_started.elapsed();
    assert_eq!(grid, GridSize { h: 24, w: 24 });
    assert_eq!(rows.len(), 576 * 5120);
    assert!(
        rows.iter().all(|v| v.is_finite()),
        "768x768: non-finite value"
    );
    eprintln!(
        "real 768x768 (576 merged tokens): encode wall time {:.2} s on {} Rayon threads",
        encode_time.as_secs_f64(),
        rayon::current_num_threads()
    );

    let duration = started.elapsed();
    record_executed_timed(Capability::Bonsai2Mmproj, REAL_TEST, duration);
    report_capability(&format!(
        "CAPABILITY-REPORT capability={} executed=true test={REAL_TEST} duration_ms={} \
         (64x64 f64 err {err_small:.3e}, 256x192 f64 err {err_wide:.3e}, \
         768x768 encode {:.2} s)",
        Capability::Bonsai2Mmproj,
        duration.as_millis(),
        encode_time.as_secs_f64()
    ));
}

/// The synthetic projector through the public API only, against the
/// path-included `f64` reference: runs everywhere, with or without the real
/// file.
#[test]
fn synthetic_mmproj_matches_the_f64_reference_through_the_public_api() {
    let bytes = synthetic_mmproj_gguf(&MmprojFixtureSpec::tiny()).expect("synthetic projector");
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let tower = VisionTower::from_mmproj(&gguf).expect("load");
    for (w, h) in [(64, 64), (96, 64)] {
        let img = image(w, h);
        let (rows, grid) = tower.encode(&img, 64).expect("encode");
        let reference = f64_reference::encode_rgb8(&gguf, w, h, &img.data).expect("f64 reference");
        assert_eq!((grid.h, grid.w), (reference.grid_h, reference.grid_w));
        let err = f64_reference::max_row_relative_error(&rows, &reference.rows, reference.dim);
        assert!(err <= F64_BOUND, "synthetic {w}x{h}: {err:.3e}");
    }
}

/// Text-only inference needs no projector, and loading and running one in
/// the same process leaves a text model's logits bit-identical: the vision
/// tower keeps no global state the text path could observe.
#[test]
fn text_only_inference_is_unchanged_by_the_vision_tower() {
    let text = tiny_dense_qwen3_gguf(0x7E57).expect("tiny dense qwen3 fixture");
    let text_gguf = GgufFile::parse(&text).expect("parse the text model");
    let kernel = KernelDispatcher::with_tier(KernelTier::Reference);
    let text_logits = || {
        let mut model = BonsaiModel::from_gguf(&text_gguf, 32).expect("load the text model");
        let mut all = Vec::new();
        for (pos, token) in [3u32, 1, 4, 1, 5].into_iter().enumerate() {
            all.extend(model.forward(token, pos, &kernel).expect("forward"));
        }
        all
    };

    let before = text_logits();
    assert!(before.iter().all(|v| v.is_finite()));

    // A text GGUF is never mistaken for a projector...
    assert!(VisionTower::from_mmproj(&text_gguf).is_err());
    // ...and running a projector changes nothing on the text side.
    let bytes = synthetic_mmproj_gguf(&MmprojFixtureSpec::tiny()).expect("synthetic projector");
    let mmproj = GgufFile::parse(&bytes).expect("parse the projector");
    let tower = VisionTower::from_mmproj(&mmproj).expect("load the projector");
    let (rows, _) = tower.encode(&image(64, 64), 16).expect("encode");
    assert!(rows.iter().all(|v| v.is_finite()));

    let after = text_logits();
    assert_eq!(
        before, after,
        "text logits changed after the vision tower ran"
    );
}
