//! Real-weight gate of the Metal vision tower (bonsai2-design.md §6): the
//! Bonsai 2 projector on the GPU against the CPU tower and the `f64`
//! reference of the same graph, and its encode time.
//!
//! # What is checked
//!
//! * The Metal tower binds all 27 blocks (334 tensors) and keeps resident
//!   exactly what `VisionTowerMetal::footprint` reads from the file (its
//!   weights in the file's own `Q8_0` / `F16` storage, its scratch and the
//!   host position grid).
//! * **Rows against the CPU tower** on the vendored 256 x 192 golden image
//!   and a synthetic 768 x 768 image: the same merged grid, every row's
//!   cosine similarity at least 0.9999, its relative L2 error at most
//!   3e-3 and its largest absolute difference at most
//!   5e-3 — both towers compute with the same weights (the
//!   Metal tower reads the `Q8_0` blocks exactly), so what remains is `f32`
//!   summation order; the measured deviation is printed next to each bound.
//! * **Rows against the `f64` reference** (64 x 64 and 256 x 192): the
//!   Metal tower's largest per-row relative error is within five
//!   times the CPU tower's own.
//! * **Time**: the best of two encodes of the 768 x 768 image (576 merged
//!   tokens) within 6 s and of the 256 x 192 image within
//!   0.75 s (the CPU tower takes 26–30 s and 4.6 s).
//!
//! # The file
//!
//! `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` is found only through
//! `$OXI_BONSAI2_MMPROJ_GGUF` or `$OXIBONSAI_MODELS_DIR`. When it is absent
//! (or the build or host has no Metal) the test records a `bonsai2-mmproj`
//! / `executed: false` capability line and returns — unless
//! `OXI_REQUIRE_MODEL_FILES=1`, which turns the absence into a failure.
//! `executed: true` is recorded only after every assertion has passed.

#[path = "../src/vision/f64_reference.rs"]
mod f64_reference;

use std::path::PathBuf;

use oxibonsai_testkit::capability::{record_skipped, Capability};

/// Names the real projector file.
const MMPROJ_ENV: &str = "OXI_BONSAI2_MMPROJ_GGUF";
/// A directory holding the release files under their own names.
const MODELS_DIR_ENV: &str = "OXIBONSAI_MODELS_DIR";
/// Release file name of the real projector.
const MMPROJ_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";
/// `1` turns a missing model file into a failure.
const REQUIRE_ENV: &str = "OXI_REQUIRE_MODEL_FILES";
const REAL_TEST: &str = "oxibonsai-model::vision_metal_tests::\
                         real_mmproj_metal_tower_tracks_the_cpu_tower_and_the_f64_reference";

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

/// The projector from `$OXI_BONSAI2_MMPROJ_GGUF` or `$OXIBONSAI_MODELS_DIR`
/// only — never a hard-coded or workspace path.
fn locate_mmproj() -> Option<PathBuf> {
    env_path(MMPROJ_ENV)
        .or_else(|| env_path(MODELS_DIR_ENV).map(|dir| dir.join(MMPROJ_FILE)))
        .filter(|p| p.is_file())
}

fn require_model_files() -> bool {
    std::env::var(REQUIRE_ENV).is_ok_and(|v| v.trim() == "1")
}

/// A line on the process's standard error that libtest does not capture,
/// so the evidence shows up in every run's log.
fn report(line: &str) {
    use std::io::Write as _;
    let mut stderr = std::io::stderr().lock();
    // Bookkeeping only: a failed diagnostic write must not fail the test.
    let _ = writeln!(stderr, "{line}");
}

/// Skip (with a capability record) or, under `OXI_REQUIRE_MODEL_FILES=1`,
/// fail.
fn skip_real(why: &str) {
    assert!(
        !require_model_files(),
        "{REAL_TEST}: {why} and {REQUIRE_ENV}=1"
    );
    record_skipped(Capability::Bonsai2Mmproj, REAL_TEST);
    report(&format!(
        "CAPABILITY-REPORT capability={} executed=false test={REAL_TEST} ({why})",
        Capability::Bonsai2Mmproj
    ));
}

/// Per-row agreement of `got` with `want` (`dim`-wide rows): the smallest
/// cosine, the largest relative L2 error and the largest absolute
/// difference.
fn row_agreement(got: &[f32], want: &[f32], dim: usize) -> (f64, f64, f32) {
    assert_eq!(got.len(), want.len());
    let mut worst_cos = f64::INFINITY;
    let mut worst_rel = 0.0f64;
    let mut worst_abs = 0.0f32;
    for (g, w) in got.chunks(dim).zip(want.chunks(dim)) {
        let (mut dot, mut ng, mut nw, mut diff) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in g.iter().zip(w) {
            let (a64, b64) = (f64::from(a), f64::from(b));
            dot += a64 * b64;
            ng += a64 * a64;
            nw += b64 * b64;
            diff += (a64 - b64) * (a64 - b64);
            worst_abs = worst_abs.max((a - b).abs());
        }
        worst_cos = worst_cos.min(dot / (ng.sqrt() * nw.sqrt()).max(f64::MIN_POSITIVE));
        worst_rel = worst_rel.max(diff.sqrt() / nw.sqrt().max(f64::MIN_POSITIVE));
    }
    (worst_cos, worst_rel, worst_abs)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod real {
    use super::*;

    use std::time::{Duration, Instant};

    use oxibonsai_core::gguf::reader::{mmap_gguf_file, GgufFile};
    use oxibonsai_model::vision::metal::VisionTowerMetal;
    use oxibonsai_model::vision::{
        decode_image, GridSize, ImageRgb8, VisionTower, DEFAULT_IMAGE_MAX_TOKENS,
    };
    use oxibonsai_testkit::capability::record_executed_timed;
    use oxibonsai_testkit::mmproj_fixture::pattern_rgb8;

    /// Every row's cosine similarity to the CPU tower's row.
    const COS_FLOOR: f64 = 0.9999;
    /// Every row's relative L2 error against the CPU tower's row.
    const REL_L2_BOUND: f64 = 3e-3;
    /// Every element's absolute difference from the CPU tower.
    const MAX_ABS_BOUND: f32 = 5e-3;
    /// The Metal tower's error against `f64` within this many times the CPU
    /// tower's own (floored at 1e-6, as the hermetic gate is).
    const F64_FACTOR: f64 = 5.0;
    /// The 768 x 768 encode (576 merged tokens), best of two, in seconds.
    const MAX_SECONDS_768: f64 = 6.0;
    /// The 256 x 192 encode (48 merged tokens), best of two, in seconds.
    const MAX_SECONDS_256: f64 = 0.75;

    fn metal_available() -> bool {
        match oxibonsai_kernels::MetalGraph::shared_device() {
            Ok(_) => true,
            Err(oxibonsai_kernels::MetalGraphError::DeviceNotFound) => false,
            Err(e) => panic!("the Metal device must open on this host: {e}"),
        }
    }

    fn golden_fixture() -> ImageRgb8 {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/bonsai2_golden_vision/fixture_256x192.png");
        let png = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        decode_image(&png).expect("the golden fixture decodes")
    }

    fn pattern(width: usize, height: usize) -> ImageRgb8 {
        ImageRgb8::new(width, height, pattern_rgb8(width, height)).expect("pattern image")
    }

    /// Encode twice and keep the faster run (the first also builds the
    /// image size's position rows and grows the scratch).
    fn encode_best_of_two(
        tower: &VisionTowerMetal,
        image: &ImageRgb8,
    ) -> (Vec<f32>, GridSize, Duration, Duration) {
        let t0 = Instant::now();
        let (first_rows, first_grid) = tower
            .encode(image, DEFAULT_IMAGE_MAX_TOKENS)
            .expect("metal encode");
        let first = t0.elapsed();
        let t1 = Instant::now();
        let (rows, grid) = tower
            .encode(image, DEFAULT_IMAGE_MAX_TOKENS)
            .expect("metal encode");
        let second = t1.elapsed();
        assert_eq!(grid, first_grid);
        assert_eq!(
            rows, first_rows,
            "two encodes of the same image are bit-identical"
        );
        (rows, grid, first.min(second), first)
    }

    fn load_average() -> String {
        std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
    }

    pub(super) fn run(path: &std::path::Path) {
        let started = Instant::now();
        let mmap = mmap_gguf_file(path).expect("map the projector");
        let gguf = GgufFile::parse(&mmap).expect("parse the projector");
        report(&format!(
            "bonsai2-mmproj metal: {} ({} bytes); load average {}",
            path.display(),
            mmap.len(),
            load_average()
        ));

        let loading = Instant::now();
        let metal = VisionTowerMetal::from_mmproj(&gguf, DEFAULT_IMAGE_MAX_TOKENS)
            .expect("the Metal tower loads");
        let metal_load = loading.elapsed();
        assert_eq!(metal.block_count(), 27, "ViT blocks bound");
        assert_eq!(gguf.tensors.len(), 334);
        assert_eq!(metal.bound_tensor_count(), gguf.tensors.len());
        let resident = metal.resident_bytes() as u64;
        let footprint =
            VisionTowerMetal::footprint(&gguf, DEFAULT_IMAGE_MAX_TOKENS).expect("footprint");
        assert_eq!(resident, footprint, "the footprint is what the tower keeps");
        let cpu_footprint = VisionTower::footprint(metal.config());
        report(&format!(
            "bonsai2-mmproj metal: loaded in {:.2}s; resident {resident} bytes ({:.3} GiB: Q8_0 \
             and F16 weights as stored, scratch for {DEFAULT_IMAGE_MAX_TOKENS}-token images, host \
             position grid) vs the CPU tower's {cpu_footprint} bytes ({:.3} GiB of f32)",
            metal_load.as_secs_f64(),
            resident as f64 / f64::from(1u32 << 30),
            cpu_footprint as f64 / f64::from(1u32 << 30)
        ));
        assert!(resident < cpu_footprint);

        let loading = Instant::now();
        let cpu = VisionTower::from_mmproj(&gguf).expect("the CPU tower loads");
        assert_eq!(cpu.resident_bytes() as u64, cpu_footprint);
        report(&format!(
            "bonsai2-mmproj cpu tower loaded in {:.2}s",
            loading.elapsed().as_secs_f64()
        ));
        let dim = metal.config().projection_dim;

        // Rows against the CPU tower, and the time of each encode. Every
        // measurement is taken (and printed) before any bound is checked.
        let mut failures = Vec::new();
        let mut worst = (f64::INFINITY, 0.0f64, 0.0f32);
        let mut timings = Vec::new();
        for (label, image, grid, bound) in [
            (
                "256x192 golden fixture",
                golden_fixture(),
                GridSize { h: 6, w: 8 },
                MAX_SECONDS_256,
            ),
            (
                "768x768 synthetic",
                pattern(768, 768),
                GridSize { h: 24, w: 24 },
                MAX_SECONDS_768,
            ),
        ] {
            let (rows, got_grid, best, first) = encode_best_of_two(&metal, &image);
            assert_eq!(got_grid, grid, "{label}");
            let cpu_started = Instant::now();
            let (want, want_grid) = cpu
                .encode(&image, DEFAULT_IMAGE_MAX_TOKENS)
                .expect("cpu encode");
            let cpu_time = cpu_started.elapsed();
            assert_eq!(want_grid, grid, "{label}");
            assert!(rows.iter().all(|v| v.is_finite()), "{label}: non-finite");
            let (cos, rel, abs) = row_agreement(&rows, &want, dim);
            report(&format!(
                "bonsai2-mmproj metal {label}: {} x {} grid; vs the CPU tower worst row cosine \
                 {cos:.8}, worst row rel-L2 {rel:.3e}, max |d| {abs:.3e}; encode best {:.3}s \
                 (first {:.3}s; gate {bound}s) vs CPU {:.2}s ({:.1}x)",
                grid.h,
                grid.w,
                best.as_secs_f64(),
                first.as_secs_f64(),
                cpu_time.as_secs_f64(),
                cpu_time.as_secs_f64() / best.as_secs_f64().max(1e-9)
            ));
            worst = (worst.0.min(cos), worst.1.max(rel), worst.2.max(abs));
            timings.push((label, best.as_secs_f64(), bound));
            if cos < COS_FLOOR {
                failures.push(format!("{label}: worst row cosine {cos} < {COS_FLOOR}"));
            }
            if rel > REL_L2_BOUND {
                failures.push(format!(
                    "{label}: worst row rel-L2 {rel:.3e} > {REL_L2_BOUND:e}"
                ));
            }
            if abs > MAX_ABS_BOUND {
                failures.push(format!("{label}: max |d| {abs:.3e} > {MAX_ABS_BOUND:e}"));
            }
        }

        // Rows against the f64 reference of the same graph.
        let mut f64_errors = Vec::new();
        for (label, image) in [("64x64", pattern(64, 64)), ("256x192", golden_fixture())] {
            let (rows, grid) = metal
                .encode(&image, DEFAULT_IMAGE_MAX_TOKENS)
                .expect("metal encode");
            let (cpu_rows, _) = cpu
                .encode(&image, DEFAULT_IMAGE_MAX_TOKENS)
                .expect("cpu encode");
            let reference =
                f64_reference::encode_rgb8(&gguf, image.width, image.height, &image.data)
                    .expect("f64 reference");
            assert_eq!((reference.grid_h, reference.grid_w), (grid.h, grid.w));
            let metal_err =
                f64_reference::max_row_relative_error(&rows, &reference.rows, reference.dim);
            let cpu_err =
                f64_reference::max_row_relative_error(&cpu_rows, &reference.rows, reference.dim);
            let bound = F64_FACTOR * cpu_err.max(1e-6);
            report(&format!(
                "bonsai2-mmproj metal {label}: max per-row relative error vs f64: metal \
                 {metal_err:.3e}, cpu {cpu_err:.3e} (bound {bound:.3e})"
            ));
            if metal_err > bound {
                failures.push(format!(
                    "{label}: metal {metal_err:.3e} vs f64 exceeds {bound:.3e} (cpu {cpu_err:.3e})"
                ));
            }
            f64_errors.push((label, metal_err, cpu_err));
        }

        for (label, best, bound) in &timings {
            if best > bound {
                failures.push(format!(
                    "{label}: encode {best:.3}s exceeds the {bound}s gate (load average {})",
                    load_average()
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("; "));

        let duration = started.elapsed();
        record_executed_timed(Capability::Bonsai2Mmproj, REAL_TEST, duration);
        report(&format!(
            "CAPABILITY-REPORT capability={} executed=true test={REAL_TEST} duration_ms={} \
             (metal vs cpu: worst cos {:.8}, rel-L2 {:.3e}, max |d| {:.3e}; f64 {:?}; encode \
             {:?}; resident {resident} bytes)",
            Capability::Bonsai2Mmproj,
            duration.as_millis(),
            worst.0,
            worst.1,
            worst.2,
            f64_errors,
            timings
        ));
    }

    /// Whether the Metal tower can run here: a Metal device.
    pub(super) fn available() -> bool {
        metal_available()
    }
}

/// The real projector on the Metal GPU against the CPU tower and the `f64`
/// reference, and its encode time (see the module docs).
///
/// One test function on purpose: the CPU tower holds about 1.8 GB of `f32`
/// weights beside the Metal tower's 0.9 GB (weights and scratch).
#[test]
fn real_mmproj_metal_tower_tracks_the_cpu_tower_and_the_f64_reference() {
    let Some(path) = locate_mmproj() else {
        skip_real(&format!(
            "no {MMPROJ_FILE}: set {MMPROJ_ENV} or {MODELS_DIR_ENV}"
        ));
        return;
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if !real::available() {
            skip_real("no Metal device on this host");
            return;
        }
        real::run(&path);
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = path;
        skip_real("this build has no Metal backend");
    }
}

/// The per-row agreement helper on hand-made rows.
#[test]
fn row_agreement_reports_the_worst_row() {
    let want = [1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0];
    let got = [1.0f32, 0.0, 0.0, 1.0, 2.0, 0.01];
    let (cos, rel, abs) = row_agreement(&got, &want, 3);
    assert!((1.0 - cos) > 0.0 && (1.0 - cos) < 1e-4, "{cos}");
    assert!((rel - 0.01 / 5f64.sqrt()).abs() < 1e-6, "{rel}");
    assert!((abs - 0.01).abs() < 1e-6, "{abs}");
    let (cos, rel, abs) = row_agreement(&want, &want, 3);
    assert!((1.0 - cos).abs() < 1e-12, "{cos}");
    assert_eq!((rel, abs), (0.0, 0.0));
}
