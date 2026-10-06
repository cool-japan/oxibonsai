//! The Metal vision tower against the CPU tower and the `f64` reference on
//! synthetic projectors: exact (all-`F16`) weights to separate the kernels'
//! correctness from weight rounding, the `Q8_0` fixtures for the rounding,
//! the encoder enum's surface, and the refusals.

use oxibonsai_core::gguf::reader::GgufFile;
use oxibonsai_kernels::MetalGraphError;
use oxibonsai_testkit::gguf_fixture::FixtureQuant;
use oxibonsai_testkit::mmproj_fixture::{
    assemble, metadata, pattern_rgb8, tensors, MmprojFixtureSpec,
};

use super::*;
use crate::vision::f64_reference as reference;
use crate::vision::{VisionEncoder, VisionTower};

/// `false` only on a host without a Metal device.
fn metal_available() -> bool {
    match oxibonsai_kernels::gpu_backend::metal_graph::MetalGraph::global() {
        Ok(_) => true,
        Err(MetalGraphError::DeviceNotFound) => false,
        Err(e) => panic!("the combined Metal library must build on this device: {e}"),
    }
}

/// The d72 projector: the real tower's head width (72) at a small scale —
/// hidden 288 (the least common multiple of 32 and 72, so every matrix is
/// a legal `Q8_0` width), 4 heads, an FFN of 400 (not a multiple of the
/// GEMM's 32-wide slice), a 1152-wide merger, 2 blocks.
fn d72_spec() -> MmprojFixtureSpec {
    MmprojFixtureSpec {
        hidden: 288,
        heads: 4,
        ffn: 400,
        blocks: 2,
        patch: 16,
        pos_side: 8,
        merger_hidden: 1152,
        projection_dim: 96,
        seed: 0x00D7_2F16,
    }
}

/// `spec`'s projector with every `Q8_0` matrix written as `F16` instead, so
/// the device's `f16` weights are exactly the CPU tower's.
fn all_f16(spec: &MmprojFixtureSpec) -> Vec<u8> {
    let mut ts = tensors(spec);
    for t in &mut ts {
        if t.quant == FixtureQuant::Q8_0 {
            t.quant = FixtureQuant::F16;
        }
    }
    assemble(&metadata(spec), &ts).expect("assemble the all-F16 projector")
}

fn native(spec: &MmprojFixtureSpec) -> Vec<u8> {
    assemble(&metadata(spec), &tensors(spec)).expect("assemble the projector")
}

fn image(width: usize, height: usize) -> ImageRgb8 {
    ImageRgb8::new(width, height, pattern_rgb8(width, height)).expect("pattern image")
}

/// Per-row cosine (worst) and `max |a − b| / max |b|` of two row sets.
fn compare(a: &[f32], b: &[f32], dim: usize) -> (f64, f64) {
    let mut worst_cos = 1.0f64;
    for (ra, rb) in a.chunks(dim).zip(b.chunks(dim)) {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in ra.iter().zip(rb) {
            dot += f64::from(x) * f64::from(y);
            na += f64::from(x) * f64::from(x);
            nb += f64::from(y) * f64::from(y);
        }
        let cos = if na == 0.0 && nb == 0.0 {
            1.0
        } else {
            dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
        };
        worst_cos = worst_cos.min(cos);
    }
    let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale;
    (worst_cos, f64::from(worst))
}

/// Both towers of one projector on one image: `(cpu rows, metal rows,
/// grid)`, the grids checked equal.
fn both(bytes: &[u8], width: usize, height: usize) -> (Vec<f32>, Vec<f32>, GridSize) {
    let gguf = GgufFile::parse(bytes).expect("parse");
    let cpu = VisionTower::from_mmproj(&gguf).expect("cpu tower");
    let metal = VisionTowerMetal::from_mmproj(&gguf, 64).expect("metal tower");
    let img = image(width, height);
    let (a, ga) = cpu.encode(&img, 64).expect("cpu encode");
    let (b, gb) = metal.encode(&img, 64).expect("metal encode");
    assert_eq!(ga, gb, "the merged grid");
    assert_eq!(a.len(), b.len());
    (a, b, ga)
}

/// Exact weights (every matrix `F16` in the file, so the device's `f16`
/// copy is the CPU's dequantised weight bit for bit): the Metal rows track
/// the CPU rows to a per-row cosine of 0.999999 and a worst relative error
/// of 1e-4, on a 64 × 64 image and a 96 × 160 one (an odd 3 × 5 merged
/// grid); against the `f64` reference the Metal tower stays within five
/// times the CPU tower's own error (floored at 1e-6).
#[test]
fn metal_tower_matches_the_cpu_tower_with_exact_weights() {
    if !metal_available() {
        return;
    }
    for spec in [d72_spec(), MmprojFixtureSpec::tiny()] {
        let bytes = all_f16(&spec);
        let gguf = GgufFile::parse(&bytes).expect("parse");
        for (w, h) in [(64usize, 64usize), (160, 96)] {
            let (cpu, metal, grid) = both(&bytes, w, h);
            let (cos, rel) = compare(&metal, &cpu, spec.projection_dim);
            eprintln!(
                "exact weights hidden {} {w}x{h} ({}x{} grid): worst row cosine {cos:.9}, worst \
                 relative error {rel:.3e}",
                spec.hidden, grid.h, grid.w
            );
            assert!(cos >= 0.999_999, "{w}x{h}: row cosine {cos}");
            assert!(rel <= 1e-4, "{w}x{h}: worst relative error {rel:e}");
            let img = image(w, h);
            let reference = reference::encode_rgb8(&gguf, w, h, &img.data).expect("f64 reference");
            let cpu_err = reference::max_row_relative_error(&cpu, &reference.rows, reference.dim);
            let metal_err =
                reference::max_row_relative_error(&metal, &reference.rows, reference.dim);
            eprintln!(
                "  vs the f64 reference: CPU {cpu_err:.3e}, Metal {metal_err:.3e} (row rel-L2)"
            );
            assert!(
                metal_err <= 5.0 * cpu_err.max(1e-6),
                "{w}x{h}: Metal {metal_err:e} vs CPU {cpu_err:e} against the f64 reference"
            );
        }
    }
}

/// The file's own `Q8_0` weights, read exactly on the device (the blocks
/// as they are, never rounded to `f16`): the rows match the CPU tower as
/// closely as with exact `F16` weights (worst row cosine ≥ 0.999999, row
/// relative L2 and worst relative error ≤ 1e-4) on the tiny and the d72
/// projectors, and against the `f64` reference the Metal tower stays within
/// five times the CPU tower's own error.
#[test]
fn metal_tower_matches_the_cpu_tower_with_q8_0_weights() {
    if !metal_available() {
        return;
    }
    for spec in [MmprojFixtureSpec::tiny(), d72_spec()] {
        let bytes = native(&spec);
        let gguf = GgufFile::parse(&bytes).expect("parse");
        for (w, h) in [(64usize, 64usize), (128, 96)] {
            let (cpu, metal, _) = both(&bytes, w, h);
            let (cos, rel) = compare(&metal, &cpu, spec.projection_dim);
            let l2 = metal
                .chunks(spec.projection_dim)
                .zip(cpu.chunks(spec.projection_dim))
                .map(|(a, b)| {
                    let diff: f64 = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| f64::from(x - y).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let norm: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum::<f64>().sqrt();
                    diff / norm.max(f64::MIN_POSITIVE)
                })
                .fold(0.0f64, f64::max);
            let img = image(w, h);
            let reference = reference::encode_rgb8(&gguf, w, h, &img.data).expect("f64 reference");
            let cpu_err = reference::max_row_relative_error(&cpu, &reference.rows, reference.dim);
            let metal_err =
                reference::max_row_relative_error(&metal, &reference.rows, reference.dim);
            eprintln!(
                "Q8_0 weights hidden {} {w}x{h}: worst row cosine {cos:.9}, row rel-L2 {l2:.3e}, \
                 worst relative error {rel:.3e}; vs f64: CPU {cpu_err:.3e}, Metal {metal_err:.3e}",
                spec.hidden
            );
            assert!(cos >= 0.999_999, "{w}x{h}: row cosine {cos}");
            assert!(l2 <= 1e-4, "{w}x{h}: row relative L2 {l2:e}");
            assert!(rel <= 1e-4, "{w}x{h}: worst relative error {rel:e}");
            assert!(
                metal_err <= 5.0 * cpu_err.max(1e-6),
                "{w}x{h}: Metal {metal_err:e} vs CPU {cpu_err:e} against the f64 reference"
            );
        }
    }
}

/// The encoder enum serves both towers through the CPU tower's surface:
/// same configuration, block and tensor counts, merged grid and row shape,
/// and matching rows.
#[test]
fn the_encoder_enum_has_the_cpu_towers_surface() {
    let bytes = all_f16(&MmprojFixtureSpec::tiny());
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let cpu = VisionEncoder::Cpu(VisionTower::from_mmproj(&gguf).expect("cpu tower"));
    assert_eq!(cpu.backend_name(), "cpu");
    assert!(!cpu.is_metal());
    if !metal_available() {
        return;
    }
    let metal = VisionEncoder::Metal(VisionTowerMetal::from_mmproj(&gguf, 16).expect("metal"));
    assert_eq!(metal.backend_name(), "metal");
    assert!(metal.is_metal());
    assert_eq!(metal.config(), cpu.config());
    assert_eq!(metal.block_count(), cpu.block_count());
    assert_eq!(metal.bound_tensor_count(), cpu.bound_tensor_count());
    assert!(metal.resident_bytes() > 0 && cpu.resident_bytes() > 0);
    for (w, h) in [(64usize, 96usize), (128, 32)] {
        assert_eq!(
            metal.merged_grid(w, h, 16).expect("grid"),
            cpu.merged_grid(w, h, 16).expect("grid")
        );
        let img = image(w, h);
        let (a, ga) = cpu.encode(&img, 16).expect("cpu");
        let (b, gb) = metal.encode(&img, 16).expect("metal");
        assert_eq!(ga, gb);
        let (cos, _) = compare(&b, &a, cpu.config().projection_dim);
        assert!(cos >= 0.999_999, "{w}x{h}: {cos}");
        let planar = crate::vision::patch_embed::normalize_planar(
            &img,
            cpu.config().image_mean,
            cpu.config().image_std,
        );
        let (c, gc) = metal
            .encode_normalized(&planar, w, h, 16)
            .expect("normalised");
        assert_eq!(gc, gb);
        assert_eq!(c, b, "encode and encode_normalized agree bit for bit");
    }
}

/// The footprint computed from the configuration is what a built tower
/// holds, the per-grid host rows are cached, and every request the CPU
/// tower refuses the Metal tower refuses too — plus budgets it cannot hold.
#[test]
fn the_metal_tower_accounts_for_itself_and_refuses_bad_requests() {
    if !metal_available() {
        return;
    }
    let spec = MmprojFixtureSpec::tiny();
    let bytes = native(&spec);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let tower = VisionTowerMetal::from_mmproj(&gguf, 32).expect("metal tower");
    assert_eq!(tower.max_tokens(), 32);
    assert_eq!(
        VisionTowerMetal::footprint(&gguf, 32).expect("footprint"),
        tower.resident_bytes() as u64
    );
    assert!(
        VisionTowerMetal::footprint(&gguf, 64).expect("footprint")
            > VisionTowerMetal::footprint(&gguf, 32).expect("footprint")
    );
    // The file's storage decides the weight bytes: the same projector with
    // every matrix in F16 holds more than with its Q8_0 blocks.
    let f16_bytes = all_f16(&spec);
    let f16_gguf = GgufFile::parse(&f16_bytes).expect("parse");
    assert!(
        VisionTowerMetal::footprint(&f16_gguf, 32).expect("footprint")
            > VisionTowerMetal::footprint(&gguf, 32).expect("footprint")
    );
    assert_eq!(tower.bound_tensor_count(), 12 * spec.blocks + 10);
    let img = image(64, 64);
    tower.encode(&img, 32).expect("encode");
    let first_gpu = tower.last_gpu_seconds();
    assert!(first_gpu > 0.0);
    tower
        .encode(&img, 32)
        .expect("encode again, cached grid rows");
    // Geometry the CPU tower refuses.
    assert!(
        tower.merged_grid(48, 64, 32).is_err(),
        "48 is not a multiple of 32"
    );
    assert!(tower.merged_grid(0, 64, 32).is_err());
    assert!(
        tower.encode(&image(256, 256), 32).is_err(),
        "64 tokens > 32"
    );
    assert!(
        tower.encode(&image(256, 256), 1024).is_err(),
        "the call's budget is capped at the tower's own"
    );
    // Budgets outside what a Metal tower holds.
    assert!(VisionTowerMetal::from_mmproj(&gguf, 0).is_err());
    assert!(VisionTowerMetal::from_mmproj(&gguf, METAL_VISION_MAX_TOKENS + 1).is_err());
    // A stray tensor is refused, as the CPU loader refuses it.
    let mut ts = tensors(&spec);
    let mut extra = ts[0].clone();
    extra.name = "v.blk.0.ffn_gate.weight".to_string();
    ts.push(extra);
    let stray = assemble(&metadata(&spec), &ts).expect("assemble");
    let stray_gguf = GgufFile::parse(&stray).expect("parse");
    let err = VisionTowerMetal::from_mmproj(&stray_gguf, 16).expect_err("stray tensor");
    assert!(err.to_string().contains("ffn_gate"), "{err}");
    // Blocks of mixed storage for one matrix kind are refused by name.
    let mut ts = tensors(&spec);
    for t in &mut ts {
        if t.name == "v.blk.1.attn_out.weight" {
            t.quant = FixtureQuant::F16;
        }
    }
    let mixed = assemble(&metadata(&spec), &ts).expect("assemble");
    let mixed_gguf = GgufFile::parse(&mixed).expect("parse");
    let err = VisionTowerMetal::from_mmproj(&mixed_gguf, 16).expect_err("mixed storage");
    assert!(err.to_string().contains("v.blk.1.attn_out.weight"), "{err}");
    assert!(VisionTowerMetal::footprint(&mixed_gguf, 16).is_err());
}

// ─────────────────────────────────────────────────────────────────────────
//  The process footprint across many encodes
// ─────────────────────────────────────────────────────────────────────────
//
// `-[MTLCommandQueue commandBuffer]` and
// `-[MTLCommandBuffer computeCommandEncoder]` return autoreleased objects. A
// thread with no autorelease pool keeps every one of them until it exits, so
// a server thread encoding images on a tower that did not drain a pool per
// image would grow by a command buffer and its encoder with every image. The
// test below holds the process footprint flat across many encodes, in a
// child process of its own so that no other test of this binary allocates
// while it measures.

/// Set in the child process the footprint test re-runs itself in; the child
/// does the measuring.
const FOOTPRINT_CHILD_ENV: &str = "OXIBONSAI_VISION_ENCODE_FOOTPRINT_PROBE";

/// Prefix of the one line the child prints per measured phase.
const FOOTPRINT_REPORT: &str = "vision-encode-footprint:";

/// Growth one measured phase may add. An undrained command buffer and its
/// encoder cost about 1.9 KiB per encode, so [`FOOTPRINT_ENCODES`]
/// undrained encodes grow by several MiB (measured without the pool: about
/// 8.7 MiB a phase), where a pooled phase stays within about a hundred KiB.
const FOOTPRINT_GROWTH_CEILING: i64 = 1 << 20;

/// Encodes (one command buffer each) per measured phase.
const FOOTPRINT_ENCODES: usize = 4800;

/// Measured phases: two image geometries.
const FOOTPRINT_PHASES: usize = 2;

/// `TASK_VM_INFO`, the `task_info` flavor carrying `phys_footprint`.
const TASK_VM_INFO: u32 = 22;

/// `TASK_VM_INFO_REV1_COUNT`: `task_vm_info` up to and including
/// `phys_footprint`, in 32-bit words.
const TASK_VM_INFO_REV1_WORDS: usize = 38;

/// Word offset of `phys_footprint` (a 64-bit field) in `task_vm_info`.
const PHYS_FOOTPRINT_WORD: usize = 36;

// SAFETY (declarations): the Mach `task_info` call and the `mach_task_self_`
// port as `<mach/task.h>` / `<mach/mach_init.h>` declare them; both live in
// libSystem, which every macOS process links.
extern "C" {
    #[link_name = "mach_task_self_"]
    static MACH_TASK_SELF: u32;
    fn task_info(
        target_task: u32,
        flavor: u32,
        task_info_out: *mut i32,
        task_info_out_cnt: *mut u32,
    ) -> i32;
}

/// This process's `phys_footprint` (`task_info(TASK_VM_INFO)`): dirty
/// anonymous memory, compressed pages and device allocations — what the
/// kernel's memory ledger charges the process, clean file pages excluded.
fn phys_footprint_bytes() -> i64 {
    let mut words = [0i32; TASK_VM_INFO_REV1_WORDS];
    let mut count = TASK_VM_INFO_REV1_WORDS as u32;
    // SAFETY: `words` is a caller-owned buffer of `count` 32-bit words, so
    // the kernel writes only inside it; `MACH_TASK_SELF` is initialised by
    // libSystem before `main`.
    let rc = unsafe { task_info(MACH_TASK_SELF, TASK_VM_INFO, words.as_mut_ptr(), &mut count) };
    assert_eq!(rc, 0, "task_info(TASK_VM_INFO) failed: kern_return_t {rc}");
    assert!(
        count as usize >= TASK_VM_INFO_REV1_WORDS,
        "task_info(TASK_VM_INFO) returned {count} words, fewer than rev1's {TASK_VM_INFO_REV1_WORDS}"
    );
    let low = u64::from(words[PHYS_FOOTPRINT_WORD] as u32);
    let high = u64::from(words[PHYS_FOOTPRINT_WORD + 1] as u32);
    let bytes = i64::try_from(low | (high << 32)).expect("footprint fits i64");
    assert!(
        bytes > 0,
        "task_info(TASK_VM_INFO) reported a zero footprint"
    );
    bytes
}

/// `bytes` as signed MiB.
fn mib(bytes: i64) -> String {
    format!("{:+.3} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// The measuring half of
/// [`metal_tower_encodes_do_not_grow_the_process_footprint`], run in the
/// child process: per image geometry, 20 warm-up encodes (pipelines, the
/// cached grid rows, allocator high-water marks), then
/// [`FOOTPRINT_ENCODES`] measured ones.
fn encode_footprint_probe() {
    let spec = MmprojFixtureSpec::tiny();
    let bytes = native(&spec);
    let gguf = GgufFile::parse(&bytes).expect("parse");
    let tower = VisionTowerMetal::from_mmproj(&gguf, 64).expect("metal tower");
    let mut failures = Vec::new();
    for (w, h) in [(64usize, 64usize), (160, 96)] {
        let img = image(w, h);
        let label = format!("{w}x{h} encode");
        let encode = || {
            let (rows, grid) = tower.encode(&img, 64).expect("encode");
            assert_eq!(rows.len(), grid.n_tokens() * spec.projection_dim);
        };
        for _ in 0..20 {
            encode();
        }
        let before = phys_footprint_bytes();
        for _ in 0..FOOTPRINT_ENCODES {
            encode();
        }
        let after = phys_footprint_bytes();
        let growth = after - before;
        println!(
            "{FOOTPRINT_REPORT} {label}: {FOOTPRINT_ENCODES} encodes grew the process footprint \
             by {} ({before} -> {after} bytes)",
            mib(growth)
        );
        if growth > FOOTPRINT_GROWTH_CEILING {
            failures.push(format!("{label}: {}", mib(growth)));
        }
    }
    assert!(
        failures.is_empty(),
        "the Metal tower's encodes grew the process footprint past {} per phase — something \
         each encode creates outlives it (an undrained autorelease pool?): {failures:?}",
        mib(FOOTPRINT_GROWTH_CEILING)
    );
}

/// The Metal tower's encodes do not grow the process: 4800 encodes of a
/// 64 × 64 image and 4800 of a 160 × 96 one grow the process footprint by
/// at most 1 MiB each. Every encode's command buffer and encoder are
/// autoreleased objects the tower drains per image; left to the thread they
/// cost about 1.9 KiB per encode until it exits.
///
/// The measurement runs in a child process running only this test (the
/// test binary re-executed with `--exact`), since other tests of this
/// binary allocate on their own threads meanwhile.
#[test]
fn metal_tower_encodes_do_not_grow_the_process_footprint() {
    if std::env::var_os(FOOTPRINT_CHILD_ENV).is_some() {
        encode_footprint_probe();
        return;
    }
    if !metal_available() {
        return;
    }
    let path = module_path!();
    let module = path.split_once("::").map_or(path, |(_, rest)| rest);
    let name = format!("{module}::metal_tower_encodes_do_not_grow_the_process_footprint");
    let exe = std::env::current_exe().expect("the test binary's path");
    let output = std::process::Command::new(exe)
        .args([name.as_str(), "--exact", "--nocapture", "--test-threads=1"])
        .env(FOOTPRINT_CHILD_ENV, "1")
        .output()
        .expect("the footprint probe process starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // `--nocapture` lets the first report share a line with libtest's own
    // `test <name> ... ` prefix, so a report is found anywhere in a line.
    let reports: Vec<&str> = stdout
        .lines()
        .chain(stderr.lines())
        .filter_map(|line| line.find(FOOTPRINT_REPORT).map(|at| &line[at..]))
        .collect();
    for line in &reports {
        eprintln!("{line}");
    }
    assert!(
        output.status.success(),
        "the footprint probe failed ({}):\n{stdout}\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("1 passed"),
        "the probe process ran no test named {name}:\n{stdout}"
    );
    assert_eq!(
        reports.len(),
        FOOTPRINT_PHASES,
        "every measured phase reports once:\n{stdout}\n{stderr}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
//  The bounded grid-rows cache
// ─────────────────────────────────────────────────────────────────────────
//
// The tower keeps the host rows of a patch grid (resized position rows and
// rotary rows) keyed by the grid of the client's image, so the number of
// entries must not follow request input: at most `GRID_CACHE_ENTRIES` grids
// are kept, the least recently used one goes when a new one arrives, and a
// grid that was evicted is rebuilt from the tower's weights alone. Nor may
// their bytes, which follow the image: the rows held never sum to more than
// `GRID_CACHE_BYTES`, and rows that alone exceed it are not cached. The
// byte-bound tests drive that with small synthetic rows through
// `GridCache::with_limits`, so none of them allocates anything near the real
// budget.

/// The patch grid of [`wide_image`]`(k)`: `(2, 2k + 2)` — `k = 1, 2, …` are
/// distinct grids.
fn wide_grid(k: usize) -> GridKey {
    (2, 2 * k + 2)
}

/// A 32-pixel-high image `32 × (k + 1)` pixels wide (`k = 1` is 64 × 32): a
/// `1 × (k + 1)` merged grid, whose patch grid is [`wide_grid`]`(k)`.
fn wide_image(k: usize) -> ImageRgb8 {
    image(32 * (k + 1), 32)
}

/// A cache entry that the single value of its rows tells from the others.
fn marked_rows(mark: f32) -> Arc<GridRows> {
    Arc::new(GridRows {
        pos: vec![mark],
        cos: vec![mark],
        sin: vec![mark],
    })
}

/// The bit patterns of `values`: equality on these is bit-identity (`==` on
/// floats equates `0.0` with `-0.0` and refuses a NaN its own copy).
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Bytes the three row sets of `rows` hold.
fn rows_bytes(rows: &GridRows) -> usize {
    std::mem::size_of_val(rows.pos.as_slice())
        + std::mem::size_of_val(rows.cos.as_slice())
        + std::mem::size_of_val(rows.sin.as_slice())
}

/// The tiny projector's tower, for images of up to 64 merged tokens.
fn tiny_tower() -> VisionTowerMetal {
    let bytes = native(&MmprojFixtureSpec::tiny());
    let gguf = GgufFile::parse(&bytes).expect("parse");
    VisionTowerMetal::from_mmproj(&gguf, 64).expect("metal tower")
}

/// The grids the tower's cache holds, least recently used first.
fn cached_grids(tower: &VisionTowerMetal) -> Vec<GridKey> {
    tower.lock_grids().expect("the grid cache lock").keys()
}

/// Rows of exactly `bytes` bytes (a whole number of `f32` words), every value
/// `mark`, spread over all three row sets so that a size function that leaves
/// one of them out is caught.
fn sized_rows(bytes: usize, mark: f32) -> Arc<GridRows> {
    let word = std::mem::size_of::<f32>();
    assert_eq!(bytes % word, 0, "whole f32 words");
    let words = bytes / word;
    let third = words / 3;
    Arc::new(GridRows {
        pos: vec![mark; words - 2 * third],
        cos: vec![mark; third],
        sin: vec![mark; third],
    })
}

/// The cache's invariant after any operation: at most `max_entries` entries,
/// one per grid; at most `max_bytes` bytes; and the incrementally tracked
/// total is what a fresh sum over the entries gives — taken here with the
/// test's own [`rows_bytes`], not the cache's size function.
fn assert_cache_invariant(cache: &GridCache, context: &str) {
    let recomputed: usize = cache.entries.iter().map(|(_, rows)| rows_bytes(rows)).sum();
    assert_eq!(
        cache.held_bytes, recomputed,
        "{context}: the tracked total against a fresh sum over the entries"
    );
    assert!(
        cache.entries.len() <= cache.max_entries,
        "{context}: {} entries against the bound of {}",
        cache.entries.len(),
        cache.max_entries
    );
    assert!(
        cache.held_bytes <= cache.max_bytes,
        "{context}: {} bytes against the bound of {}",
        cache.held_bytes,
        cache.max_bytes
    );
    let mut keys = cache.keys();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(keys.len(), cache.entries.len(), "{context}: one per grid");
}

/// A 64-bit linear congruential generator: reproducible pseudo-random
/// sequences for the mixed-sequence tests, with no dependency.
struct Lcg(u64);

impl Lcg {
    fn next_word(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// A value in `0..n` (`n` non-zero).
    fn below(&mut self, n: usize) -> usize {
        let n = u64::try_from(n).expect("a bound that fits u64");
        usize::try_from(self.next_word() % n).expect("a value below a usize bound")
    }
}

/// What the cache's rules make of one insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InsertOutcome {
    /// The key was cached: its entry is kept and refreshed.
    Duplicate,
    /// The rows alone exceed the byte bound: not cached, nothing evicted.
    Refused,
    /// Cached after evicting this many least recently used entries.
    Cached(usize),
}

/// The cache's rules written as plainly as possible — `(key, bytes)` pairs,
/// least recently used first, with the sizes the test asked for rather than
/// anything measured from the rows — to hold the real cache to after every
/// operation of a mixed sequence.
struct CacheModel {
    entries: Vec<(GridKey, usize)>,
    max_entries: usize,
    max_bytes: usize,
}

impl CacheModel {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
            max_bytes,
        }
    }

    fn total(&self) -> usize {
        self.entries.iter().map(|(_, bytes)| bytes).sum()
    }

    fn keys(&self) -> Vec<GridKey> {
        self.entries.iter().map(|(key, _)| *key).collect()
    }

    /// A lookup: a hit makes the entry the most recently used.
    fn get(&mut self, key: GridKey) -> bool {
        match self.entries.iter().position(|(k, _)| *k == key) {
            Some(at) => {
                let entry = self.entries.remove(at);
                self.entries.push(entry);
                true
            }
            None => false,
        }
    }

    fn insert(&mut self, key: GridKey, bytes: usize) -> InsertOutcome {
        if self.get(key) {
            return InsertOutcome::Duplicate;
        }
        if bytes > self.max_bytes {
            return InsertOutcome::Refused;
        }
        let mut evicted = 0;
        while self.entries.len() + 1 > self.max_entries || self.total() + bytes > self.max_bytes {
            self.entries.remove(0);
            evicted += 1;
        }
        self.entries.push((key, bytes));
        InsertOutcome::Cached(evicted)
    }
}

/// An insert never leaves more than the bound, whatever the number of
/// distinct grids: after every insert the cache holds `min(inserts, bound)`
/// grids, the newest one the most recently used.
#[test]
fn the_grid_cache_never_holds_more_than_its_bound() {
    let mut cache = GridCache::default();
    for k in 1..=3 * GRID_CACHE_ENTRIES {
        cache.insert(wide_grid(k), marked_rows(k as f32));
        let held = cache.keys();
        assert!(
            held.len() <= GRID_CACHE_ENTRIES,
            "after {k} inserts: {held:?}"
        );
        assert_eq!(held.len(), k.min(GRID_CACHE_ENTRIES), "after {k} inserts");
        assert_eq!(
            held.last(),
            Some(&wide_grid(k)),
            "the newest grid is the most recent"
        );
    }
}

/// The survivors are the most recently used grids: a grid touched by a hit
/// outlives the one that was merely inserted first, and a miss touches
/// nothing.
#[test]
fn the_grid_cache_evicts_the_least_recently_used_grid() {
    let mut cache = GridCache::default();
    for k in 1..=GRID_CACHE_ENTRIES {
        cache.insert(wide_grid(k), marked_rows(k as f32));
    }
    // Grid 1 is the oldest; a hit makes it the newest, so grid 2 is the one
    // that goes when a ninth grid arrives.
    assert!(cache.get(wide_grid(1)).is_some());
    cache.insert(wide_grid(GRID_CACHE_ENTRIES + 1), marked_rows(0.0));
    let kept = cache.keys();
    assert_eq!(kept.len(), GRID_CACHE_ENTRIES);
    assert!(
        kept.contains(&wide_grid(1)),
        "the touched grid survived: {kept:?}"
    );
    assert!(
        !kept.contains(&wide_grid(2)),
        "the true least recently used grid was evicted: {kept:?}"
    );
    let expected: Vec<GridKey> = (3..=GRID_CACHE_ENTRIES)
        .chain([1, GRID_CACHE_ENTRIES + 1])
        .map(wide_grid)
        .collect();
    assert_eq!(kept, expected, "least recently used first");
    assert!(cache.get(wide_grid(99)).is_none());
    assert_eq!(cache.keys(), expected, "a miss changes nothing");
}

/// Two encodes of one new grid can both build it; the second insert returns
/// the entry already cached, so the grid still takes one entry.
#[test]
fn a_grid_built_twice_takes_one_entry() {
    let mut cache = GridCache::default();
    let first = marked_rows(1.0);
    cache.insert(wide_grid(1), Arc::clone(&first));
    cache.insert(wide_grid(2), marked_rows(2.0));
    let second = marked_rows(1.0);
    let kept = cache.insert(wide_grid(1), Arc::clone(&second));
    assert!(
        Arc::ptr_eq(&kept, &first),
        "the cached entry is the one returned"
    );
    assert!(!Arc::ptr_eq(&kept, &second));
    assert_eq!(
        cache.keys(),
        vec![wide_grid(2), wide_grid(1)],
        "one entry per grid, the second insert a use of it"
    );
}

/// Rows already handed out stay valid when their entry is evicted: the cache
/// lets go of them, the holder keeps them.
#[test]
fn rows_handed_out_outlive_their_eviction() {
    let mut cache = GridCache::default();
    let held = cache.insert(wide_grid(1), marked_rows(7.0));
    assert_eq!(Arc::strong_count(&held), 2, "the cache and the caller");
    for k in 2..=GRID_CACHE_ENTRIES + 1 {
        cache.insert(wide_grid(k), marked_rows(k as f32));
    }
    assert!(!cache.keys().contains(&wide_grid(1)), "grid 1 was evicted");
    assert_eq!(Arc::strong_count(&held), 1, "only the caller holds it now");
    assert_eq!((held.pos[0], held.cos[0], held.sin[0]), (7.0, 7.0, 7.0));
}

/// The figures of the cache's documentation, derived from the two constants:
/// the cache holds at most the smaller of 8 entries and 128 MiB. Bonsai 2's
/// projector (hidden 1152, head width 72) is 4 896 bytes per patch, 20.05 MB
/// for the largest grid of the default 1 024-token budget — eight of those
/// would be 160 MB, more than the byte budget, which holds six — and 321 MB
/// for the largest grid the Metal tower admits, which is more than the whole
/// budget and so never cached (eight of them were the 2.57 GB the entry bound
/// alone allowed). The largest grid that is cached is 6 853 merged tokens.
/// Then the same figures scaled down by 1024 (the budget is a power of two,
/// and so are these grids' patch counts) run through the real cache logic,
/// which needs no more than a few hundred kilobytes.
#[test]
fn the_documented_worst_case_is_the_smaller_of_eight_entries_and_128_mib() {
    assert_eq!(GRID_CACHE_ENTRIES, 8);
    assert_eq!(GRID_CACHE_BYTES, 128 * 1024 * 1024);
    assert_eq!(GRID_CACHE_BYTES, 134_217_728);
    let per_patch = (1152 + 72) * std::mem::size_of::<f32>();
    assert_eq!(per_patch, 4_896);
    // The default budget: the entry bound alone would allow 160 MB, so the
    // byte budget binds first, at six of the largest grids.
    let default_largest = 1_024 * 4 * per_patch;
    assert_eq!(default_largest, 20_054_016);
    assert_eq!(GRID_CACHE_ENTRIES * default_largest, 160_432_128);
    assert!(GRID_CACHE_ENTRIES * default_largest > GRID_CACHE_BYTES);
    assert_eq!(GRID_CACHE_BYTES / default_largest, 6);
    assert!(6 * default_largest <= GRID_CACHE_BYTES);
    assert!(7 * default_largest > GRID_CACHE_BYTES);
    // The tower's own ceiling: one grid is more than the whole budget.
    assert_eq!(METAL_VISION_MAX_TOKENS, 16_384);
    let ceiling_largest = METAL_VISION_MAX_TOKENS * 4 * per_patch;
    assert_eq!(ceiling_largest, 320_864_256);
    assert_eq!(GRID_CACHE_ENTRIES * ceiling_largest, 2_566_914_048);
    assert!(ceiling_largest > GRID_CACHE_BYTES);
    // The largest grid that fits the budget, in merged tokens.
    let cacheable_tokens = GRID_CACHE_BYTES / (4 * per_patch);
    assert_eq!(cacheable_tokens, 6_853);
    assert!(cacheable_tokens * 4 * per_patch <= GRID_CACHE_BYTES);
    assert!((cacheable_tokens + 1) * 4 * per_patch > GRID_CACHE_BYTES);
    // The tower's cache is built with exactly these limits.
    let production = GridCache::default();
    assert_eq!(production.max_entries, GRID_CACHE_ENTRIES);
    assert_eq!(production.max_bytes, GRID_CACHE_BYTES);

    // The same figures divided by 1024, through the real cache logic.
    const SCALE: usize = 1_024;
    assert_eq!(default_largest % SCALE, 0);
    assert_eq!(ceiling_largest % SCALE, 0);
    let mut cache = GridCache::with_limits(GRID_CACHE_ENTRIES, GRID_CACHE_BYTES / SCALE);
    for k in 1..=GRID_CACHE_ENTRIES {
        cache.insert(wide_grid(k), sized_rows(default_largest / SCALE, k as f32));
        assert_cache_invariant(&cache, &format!("default-budget grid {k}"));
        assert_eq!(
            cache.entries.len(),
            k.min(GRID_CACHE_BYTES / default_largest),
            "after {k} of the largest default-budget grids: the byte budget binds at six"
        );
    }
    assert_eq!(
        cache.keys(),
        (3..=GRID_CACHE_ENTRIES).map(wide_grid).collect::<Vec<_>>(),
        "the six most recent of the eight"
    );
    assert_eq!(cache.held_bytes, 6 * default_largest / SCALE);
    for k in 1..=GRID_CACHE_ENTRIES {
        let ceiling = sized_rows(ceiling_largest / SCALE, 0.0);
        let returned = cache.insert(wide_grid(100 + k), Arc::clone(&ceiling));
        assert!(Arc::ptr_eq(&returned, &ceiling), "the caller's own rows");
        assert!(
            !cache.keys().contains(&wide_grid(100 + k)),
            "a ceiling-size grid is never cached"
        );
    }
    assert_eq!(
        cache.held_bytes,
        6 * default_largest / SCALE,
        "and eight of them evicted nothing"
    );
}

/// However many shapes the tower sees — encoded or only asked for — its
/// cache holds at most the entry bound after every one, the grid just used is
/// the most recent, and the rows held are what the documented size formula
/// says: their sum, which the cache tracks exactly, stays within the smaller
/// of the entry bound's worth of the largest grids the tower admits and the
/// byte budget.
#[test]
fn the_tower_keeps_at_most_the_bound_of_grids_however_many_shapes_it_sees() {
    if !metal_available() {
        return;
    }
    let tower = tiny_tower();
    let (hidden, head_dim) = (tower.config().hidden, tower.config().head_dim);
    let per_patch = (hidden + head_dim) * std::mem::size_of::<f32>();
    for k in 1..=GRID_CACHE_ENTRIES + 4 {
        tower.encode(&wide_image(k), 64).expect("encode");
        let held = cached_grids(&tower);
        assert!(
            held.len() <= GRID_CACHE_ENTRIES,
            "after encode {k}: {held:?}"
        );
        assert_eq!(held.len(), k.min(GRID_CACHE_ENTRIES), "after encode {k}");
        assert_eq!(
            held.last(),
            Some(&wide_grid(k)),
            "the grid just encoded is the newest"
        );
    }
    for k in GRID_CACHE_ENTRIES + 5..=3 * GRID_CACHE_ENTRIES {
        let (h, w) = wide_grid(k);
        let rows = tower.grid_rows(h, w).expect("rows");
        assert_eq!(rows_bytes(&rows), h * w * per_patch, "rows of {h} x {w}");
        let held = cached_grids(&tower);
        assert_eq!(held.len(), GRID_CACHE_ENTRIES, "after rows {k}: {held:?}");
        assert_eq!(held.last(), Some(&(h, w)));
    }
    // The documented worst case: the smaller of the entry bound's worth of
    // the largest grids and the byte budget.
    let worst_case =
        (GRID_CACHE_ENTRIES * tower.max_tokens() * 4 * per_patch).min(GRID_CACHE_BYTES);
    let cache = tower.lock_grids().expect("the grid cache lock");
    let resident: usize = cache.entries.iter().map(|(_, rows)| rows_bytes(rows)).sum();
    assert!(
        resident <= worst_case,
        "{resident} bytes against {worst_case}"
    );
    assert_eq!(
        cache.held_bytes, resident,
        "the tracked total is the rows held"
    );
}

/// On the tower: a grid that is used again outlives the one that was merely
/// built first.
#[test]
fn the_tower_evicts_the_least_recently_used_grid() {
    if !metal_available() {
        return;
    }
    let tower = tiny_tower();
    for k in 1..=GRID_CACHE_ENTRIES {
        tower.encode(&wide_image(k), 64).expect("encode");
    }
    tower
        .encode(&wide_image(1), 64)
        .expect("encode grid 1 again");
    tower
        .encode(&wide_image(GRID_CACHE_ENTRIES + 1), 64)
        .expect("encode a ninth shape");
    let held = cached_grids(&tower);
    assert!(
        held.contains(&wide_grid(1)),
        "the grid used again survived: {held:?}"
    );
    assert!(
        !held.contains(&wide_grid(2)),
        "the least recently used grid went: {held:?}"
    );
    assert_eq!(held.len(), GRID_CACHE_ENTRIES);
}

/// A grid that was evicted and is asked for again is rebuilt bit for bit
/// (position, cosine and sine rows), and an encode of the same image before
/// and after its grid was evicted gives the same rows to the last bit; the
/// rows an earlier request still holds stay valid across the eviction.
#[test]
fn an_evicted_grid_is_rebuilt_bit_for_bit_and_encodes_the_same() {
    if !metal_available() {
        return;
    }
    let tower = tiny_tower();
    let (h, w) = wide_grid(1);
    let (before, _) = tower.encode(&wide_image(1), 64).expect("encode");
    let first = tower.grid_rows(h, w).expect("the cached rows");
    assert_eq!(Arc::strong_count(&first), 2, "the cache and this handle");
    // Evict it: the bound's worth of other grids, each newer than this one.
    for k in 2..=GRID_CACHE_ENTRIES + 1 {
        let (other_h, other_w) = wide_grid(k);
        tower
            .grid_rows(other_h, other_w)
            .expect("rows of another grid");
        let held = cached_grids(&tower);
        assert!(held.len() <= GRID_CACHE_ENTRIES, "after grid {k}: {held:?}");
    }
    assert!(
        !cached_grids(&tower).contains(&(h, w)),
        "the grid was evicted: {:?}",
        cached_grids(&tower)
    );
    // The handle taken before the eviction is the only one left and intact.
    assert_eq!(Arc::strong_count(&first), 1, "the cache let go of it");
    let rebuilt = tower.grid_rows(h, w).expect("the rebuilt rows");
    assert!(
        !Arc::ptr_eq(&first, &rebuilt),
        "it was built again, not found"
    );
    assert_eq!(bits(&rebuilt.pos), bits(&first.pos), "position rows");
    assert_eq!(bits(&rebuilt.cos), bits(&first.cos), "cosine rows");
    assert_eq!(bits(&rebuilt.sin), bits(&first.sin), "sine rows");
    assert!(cached_grids(&tower).contains(&(h, w)), "and cached again");
    // Evict it once more, so the encode below builds it a second time.
    for k in 2..=GRID_CACHE_ENTRIES + 1 {
        let (other_h, other_w) = wide_grid(k);
        tower
            .grid_rows(other_h, other_w)
            .expect("rows of another grid");
    }
    assert!(!cached_grids(&tower).contains(&(h, w)));
    let (after, _) = tower
        .encode(&wide_image(1), 64)
        .expect("encode after the eviction");
    assert_eq!(bits(&after), bits(&before), "the same rows to the last bit");
}

/// Shapes the concurrent tests encode (`wide_grid(1..=CONCURRENT_SHAPES)`):
/// more than the cache holds.
const CONCURRENT_SHAPES: usize = GRID_CACHE_ENTRIES + 4;

/// Encodes every one of [`CONCURRENT_SHAPES`] shapes on several threads of
/// `tower`, so grids are evicted while other encodes still use them, and
/// returns what differed from the same image encoded alone on `baseline`
/// (the tower itself, or another one): nothing, when every encode came out
/// bit-identical.
fn concurrent_encode_mismatches(
    tower: &VisionTowerMetal,
    baseline: &VisionTowerMetal,
) -> Vec<String> {
    const THREADS: usize = 6;
    const ROUNDS: usize = 3;
    let images: Vec<ImageRgb8> = (1..=CONCURRENT_SHAPES).map(wide_image).collect();
    let alone: Vec<Vec<u32>> = images
        .iter()
        .map(|img| bits(&baseline.encode(img, 64).expect("encode alone").0))
        .collect();
    let mismatches = Mutex::new(Vec::<String>::new());
    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let (images, alone, mismatches) = (&images, &alone, &mismatches);
            scope.spawn(move || {
                for round in 0..ROUNDS {
                    for step in 0..CONCURRENT_SHAPES {
                        let shape = (thread * 5 + round * 3 + step) % CONCURRENT_SHAPES;
                        let outcome = match tower.encode(&images[shape], 64) {
                            Ok((rows, _)) if bits(&rows) == alone[shape] => continue,
                            Ok(_) => "rows differ from the encode alone".to_string(),
                            Err(e) => e.to_string(),
                        };
                        mismatches.lock().expect("the mismatch list").push(format!(
                            "thread {thread} round {round} shape {shape}: {outcome}"
                        ));
                    }
                }
            });
        }
    });
    mismatches.into_inner().expect("the mismatch list")
}

/// Encodes on several threads over more shapes than the cache holds, so
/// grids are evicted while other encodes still use them: every encode comes
/// out bit-identical to the same image encoded alone, and the cache ends
/// within its bound.
#[test]
fn concurrent_encodes_over_many_shapes_stay_bit_identical_while_grids_are_evicted() {
    if !metal_available() {
        return;
    }
    let tower = tiny_tower();
    let mismatches = concurrent_encode_mismatches(&tower, &tower);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
    let held = cached_grids(&tower);
    assert!(held.len() <= GRID_CACHE_ENTRIES, "{held:?}");
}

// ─────────────────────────────────────────────────────────────────────────
//  The byte budget of the grid-rows cache
// ─────────────────────────────────────────────────────────────────────────
//
// The device-free tests drive `GridCache::with_limits` with synthetic rows of
// chosen sizes (a budget of a thousand bytes stands in for 128 MiB); the
// tower tests cut the budget of a real tiny tower to a few of its grids.

/// A cache of up to 8 entries and 1 000 bytes holding grids 1..=6 of 100 bytes
/// in order, then touched 2 and 1: least recently used first, `[3, 4, 5, 6, 2,
/// 1]`, 600 bytes held. (Six entries of eight, so only the byte bound can bind.)
fn six_grids_of_100_bytes() -> GridCache {
    let mut cache = GridCache::with_limits(8, 1_000);
    for k in 1..=6 {
        cache.insert(wide_grid(k), sized_rows(100, k as f32));
    }
    assert!(cache.get(wide_grid(2)).is_some());
    assert!(cache.get(wide_grid(1)).is_some());
    assert_eq!(cache.keys(), [3, 4, 5, 6, 2, 1].map(wide_grid).to_vec());
    assert_eq!(cache.held_bytes, 600);
    cache
}

/// The byte invariant holds after every operation of a long mixed sequence
/// of small, medium, large and over-budget entries (and lookups) over more
/// grids than the cache holds, at both bounds: the tracked total equals a
/// fresh sum over the entries, neither bound is passed, and the entries, their
/// recency order and the total are those of a plain model of the rules. The
/// sequence reaches every rule — duplicates, refusals, plain inserts and
/// inserts that evict — many times, and the rows come back the way each rule
/// says (the cached entry on a duplicate; the caller's own rows otherwise).
#[test]
fn the_grid_cache_byte_invariant_holds_after_every_insert_of_a_mixed_sequence() {
    const MAX_ENTRIES: usize = 6;
    const MAX_BYTES: usize = 1_000;
    let mut cache = GridCache::with_limits(MAX_ENTRIES, MAX_BYTES);
    let mut model = CacheModel::new(MAX_ENTRIES, MAX_BYTES);
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let (mut duplicates, mut refused, mut cached, mut evicting) = (0usize, 0usize, 0usize, 0usize);
    for step in 0..2_000usize {
        let key = wide_grid(1 + rng.below(12));
        let context = format!("step {step}, grid {key:?}");
        if rng.below(5) == 0 {
            assert_eq!(
                cache.get(key).is_some(),
                model.get(key),
                "{context}: lookup"
            );
        } else {
            // Whole words: 4..=100 bytes, 200..=596, 600..=996, and
            // 1 004..=2 000 (over the budget).
            let words = match rng.below(4) {
                0 => 1 + rng.below(25),
                1 => 50 + rng.below(100),
                2 => 150 + rng.below(100),
                _ => 251 + rng.below(250),
            };
            let bytes = words * std::mem::size_of::<f32>();
            let rows = sized_rows(bytes, step as f32);
            let returned = cache.insert(key, Arc::clone(&rows));
            match model.insert(key, bytes) {
                InsertOutcome::Duplicate => {
                    duplicates += 1;
                    assert!(
                        !Arc::ptr_eq(&returned, &rows),
                        "{context}: the cached entry"
                    );
                    assert_eq!(
                        Arc::strong_count(&rows),
                        1,
                        "{context}: the duplicate is dropped"
                    );
                }
                InsertOutcome::Refused => {
                    refused += 1;
                    assert!(bytes > MAX_BYTES, "{context}: {bytes} bytes refused");
                    assert!(
                        Arc::ptr_eq(&returned, &rows),
                        "{context}: the caller's rows"
                    );
                    assert_eq!(Arc::strong_count(&rows), 2, "{context}: no cache handle");
                }
                InsertOutcome::Cached(evictions) => {
                    cached += 1;
                    evicting += usize::from(evictions > 0);
                    assert!(
                        Arc::ptr_eq(&returned, &rows),
                        "{context}: the caller's rows"
                    );
                    assert_eq!(Arc::strong_count(&rows), 3, "{context}: one cache handle");
                }
            }
        }
        assert_cache_invariant(&cache, &context);
        assert_eq!(cache.keys(), model.keys(), "{context}: entries and order");
        assert_eq!(
            cache.held_bytes,
            model.total(),
            "{context}: the tracked total"
        );
    }
    assert!(
        duplicates >= 100 && refused >= 100 && cached >= 100 && evicting >= 100,
        "the sequence reached every rule: {duplicates} duplicates, {refused} refusals, \
         {cached} inserts, {evicting} of them evicting"
    );
}

/// Rows that alone exceed the byte budget are not cached: `insert` hands the
/// caller's own rows back (still usable), evicts nothing — not for a cache with
/// room, not for one full by entry count — and leaves the entries, their
/// recency order and the byte total exactly as they were. The boundary is
/// exact: rows of the whole budget are cached (evicting everything else), one
/// word more are not. A cache that keeps no entries caches nothing.
#[test]
fn an_entry_larger_than_the_budget_is_not_cached_and_evicts_nothing() {
    let mut cache = GridCache::with_limits(4, 1_000);
    for k in 1..=3 {
        cache.insert(wide_grid(k), sized_rows(100, k as f32));
    }
    // Touch grid 1, so that "exactly as it was" includes the recency order.
    assert!(cache.get(wide_grid(1)).is_some());
    let order = cache.keys();
    assert_eq!(order, [2, 3, 1].map(wide_grid).to_vec());

    // One word over the budget, into a cache with room for another entry.
    let over = sized_rows(1_004, 9.0);
    let returned = cache.insert(wide_grid(9), Arc::clone(&over));
    assert!(
        Arc::ptr_eq(&returned, &over),
        "the caller's own rows come back"
    );
    assert_eq!(
        Arc::strong_count(&over),
        2,
        "this handle and the returned one; the cache holds none"
    );
    assert_eq!(returned.byte_len(), 1_004);
    assert_eq!(
        (returned.pos[0], returned.cos[0], returned.sin[0]),
        (9.0, 9.0, 9.0),
        "the rows are usable"
    );
    assert_eq!(cache.keys(), order, "nothing evicted, the order untouched");
    assert_eq!(cache.held_bytes, 300);
    assert_cache_invariant(&cache, "after the refusal");
    assert!(cache.get(wide_grid(9)).is_none(), "and not cached");

    // Into a cache that is full by entry count: still nothing evicted.
    cache.insert(wide_grid(4), sized_rows(100, 4.0));
    let full = cache.keys();
    assert_eq!(full.len(), 4);
    let huge = sized_rows(4_000, 3.0);
    let returned = cache.insert(wide_grid(10), Arc::clone(&huge));
    assert!(Arc::ptr_eq(&returned, &huge));
    assert_eq!(
        cache.keys(),
        full,
        "a full cache loses nothing to a refusal"
    );
    assert_eq!(cache.held_bytes, 400);

    // The boundary: the whole budget is cached, evicting everything else ...
    let exact = sized_rows(1_000, 5.0);
    let kept = cache.insert(wide_grid(11), Arc::clone(&exact));
    assert!(Arc::ptr_eq(&kept, &exact));
    assert_eq!(
        Arc::strong_count(&exact),
        3,
        "this handle, the returned one and the cache's"
    );
    assert_eq!(cache.keys(), vec![wide_grid(11)]);
    assert_eq!(cache.held_bytes, 1_000);
    assert_cache_invariant(&cache, "after the exact fit");
    // ... and one word more is refused, leaving that entry in place.
    let refused = cache.insert(wide_grid(12), sized_rows(1_004, 6.0));
    assert_eq!(refused.byte_len(), 1_004);
    assert_eq!(cache.keys(), vec![wide_grid(11)]);
    assert_eq!(cache.held_bytes, 1_000);

    // A cache that keeps no entries caches nothing.
    let mut none = GridCache::with_limits(0, 1_000);
    let rows = sized_rows(100, 1.0);
    let returned = none.insert(wide_grid(1), Arc::clone(&rows));
    assert!(Arc::ptr_eq(&returned, &rows));
    assert_eq!(Arc::strong_count(&rows), 2, "no cache handle");
    assert!(none.keys().is_empty());
    assert_eq!(none.held_bytes, 0);
}

/// An insert that fits only after evictions drops exactly the least recently
/// used entries it needs and no more: for each size of incoming rows, the
/// evicted entries are the first `n` in recency order (touched ones outlive
/// untouched ones), `n` is the least that leaves room (one fewer would break
/// the budget), the newcomer is the most recently used, and the byte total
/// is exact — including a fit to the last byte. When the entry bound binds
/// first it drops one; when both bind it drops what the stricter needs.
#[test]
fn an_insert_that_fits_only_after_evictions_evicts_exactly_the_least_recently_used_ones_needed() {
    // Recency order of `six_grids_of_100_bytes`, least recently used first.
    let lru = [3, 4, 5, 6, 2, 1];
    let table = [
        (100, 0),
        (400, 0),
        (404, 1),
        (500, 1),
        (504, 2),
        (700, 3),
        (704, 4),
        (900, 5),
        (904, 6),
        (1_000, 6),
    ];
    for (incoming, evicted) in table {
        let context = format!("{incoming} bytes incoming");
        if evicted > 0 {
            assert!(
                600 - 100 * (evicted - 1) + incoming > 1_000,
                "{context}: evicting {} would already leave room, so {evicted} is more than needed",
                evicted - 1
            );
        }
        assert!(
            600 - 100 * evicted + incoming <= 1_000,
            "{context}: room after {evicted}"
        );
        let mut cache = six_grids_of_100_bytes();
        let rows = sized_rows(incoming, 7.0);
        let returned = cache.insert(wide_grid(7), Arc::clone(&rows));
        assert!(Arc::ptr_eq(&returned, &rows), "{context}");
        let expected: Vec<GridKey> = lru[evicted..]
            .iter()
            .copied()
            .chain([7])
            .map(wide_grid)
            .collect();
        assert_eq!(cache.keys(), expected, "{context}: survivors, newest last");
        assert_eq!(
            cache.held_bytes,
            600 - 100 * evicted + incoming,
            "{context}: the tracked total"
        );
        assert_cache_invariant(&cache, &context);
    }

    // The entry bound binds first: a full cache of small grids drops one.
    let mut cache = GridCache::with_limits(4, 1_000);
    for k in 1..=4 {
        cache.insert(wide_grid(k), sized_rows(40, k as f32));
    }
    cache.insert(wide_grid(5), sized_rows(40, 5.0));
    assert_eq!(cache.keys(), [2, 3, 4, 5].map(wide_grid).to_vec());
    assert_eq!(cache.held_bytes, 160);
    assert_cache_invariant(&cache, "entry bound alone");

    // Both bind: the entry bound takes one entry, and the bytes still need
    // another. 4 x 100 held, grid 1 touched: order 2, 3, 4, 1; then 800 in.
    let mut cache = GridCache::with_limits(4, 1_000);
    for k in 1..=4 {
        cache.insert(wide_grid(k), sized_rows(100, k as f32));
    }
    assert!(cache.get(wide_grid(1)).is_some());
    cache.insert(wide_grid(5), sized_rows(800, 5.0));
    assert_eq!(cache.keys(), [4, 1, 5].map(wide_grid).to_vec());
    assert_eq!(cache.held_bytes, 1_000);
    assert_cache_invariant(&cache, "both bounds");
}

/// Many rounds of insert, eviction and re-insert — about half of them racing
/// duplicates of a cached grid, with rows of another size — never let the
/// tracked byte total drift: it equals a fresh sum over the entries after
/// every round, a duplicate adds and releases nothing and is not stored, and
/// once the churn is flushed out by whole-budget rows the total is exactly
/// the live entries' bytes.
#[test]
fn eviction_and_reinsert_keep_the_tracked_total_exact_over_many_rounds() {
    const MAX_ENTRIES: usize = 5;
    const MAX_BYTES: usize = 1_200;
    let mut cache = GridCache::with_limits(MAX_ENTRIES, MAX_BYTES);
    let mut rng = Lcg(0x0DDB_1A5E_5BAD_5EED);
    let mut racing = 0usize;
    let mut evictions = 0usize;
    for round in 0..4_000usize {
        let key = wide_grid(1 + rng.below(9));
        let rows = sized_rows(4 * (1 + rng.below(80)), round as f32);
        assert!(
            rows.byte_len() <= MAX_BYTES,
            "every entry fits the budget alone"
        );
        let before = (cache.entries.len(), cache.held_bytes);
        let was_cached = cache.keys().contains(&key);
        let kept = cache.insert(key, Arc::clone(&rows));
        if was_cached {
            racing += 1;
            assert!(
                !Arc::ptr_eq(&kept, &rows),
                "round {round}: the cached entry wins"
            );
            assert_eq!(
                Arc::strong_count(&rows),
                1,
                "round {round}: the duplicate is dropped"
            );
            assert_eq!(
                (cache.entries.len(), cache.held_bytes),
                before,
                "round {round}: a duplicate adds and releases nothing"
            );
            assert_eq!(cache.keys().last(), Some(&key), "round {round}: refreshed");
        } else {
            // The new entry always goes in, so the count grew by one less the
            // entries evicted for it.
            let evicted = (before.0 + 1).saturating_sub(cache.entries.len());
            evictions += usize::from(evicted > 0);
        }
        assert_cache_invariant(&cache, &format!("round {round}"));
    }
    assert!(
        racing >= 500 && evictions >= 500,
        "{racing} racing duplicates and {evictions} evicting inserts over 4000 rounds"
    );
    // Flush: rows of the whole budget leave exactly their own bytes, and then
    // small rows push them out in turn, each time leaving exactly what lives.
    for cycle in 0..50usize {
        cache.insert(wide_grid(100 + cycle), sized_rows(MAX_BYTES, 0.0));
        assert_eq!(cache.keys(), vec![wide_grid(100 + cycle)], "cycle {cycle}");
        assert_eq!(
            cache.held_bytes, MAX_BYTES,
            "cycle {cycle}: nothing left over"
        );
        for k in 1..=MAX_ENTRIES {
            cache.insert(wide_grid(k), sized_rows(100, k as f32));
        }
        assert_eq!(cache.keys().len(), MAX_ENTRIES, "cycle {cycle}");
        assert_eq!(cache.held_bytes, MAX_ENTRIES * 100, "cycle {cycle}: exact");
        assert_cache_invariant(&cache, &format!("cycle {cycle}"));
    }
}

/// Bytes of the rows of `wide_grid(k)` on `tower`: `n_patches × (hidden +
/// head_dim) × 4`.
fn wide_grid_bytes(tower: &VisionTowerMetal, k: usize) -> usize {
    let (grid_h, grid_w) = wide_grid(k);
    let per_patch = (tower.config().hidden + tower.config().head_dim) * std::mem::size_of::<f32>();
    grid_h * grid_w * per_patch
}

/// On a real tower whose cache budget is cut to the rows of its ninth wide
/// grid: after every encode both bounds hold and the tracked total is what
/// the entries sum to; a grid that fits is cached as the newest, one whose
/// rows alone exceed the budget is encoded bit-identically to a tower with
/// the production limits but is not cached and evicts nothing (and is built
/// again by the next request); a budget below every grid keeps the cache empty
/// while every encode still matches; and with the production limits back, the
/// same grid caches normally. A fresh tower is built with the production
/// limits.
#[test]
fn the_tower_keeps_within_its_byte_budget_and_encodes_an_oversized_grid_uncached() {
    if !metal_available() {
        return;
    }
    let tower = tiny_tower();
    let reference = tiny_tower();
    {
        let cache = tower.lock_grids().expect("the grid cache lock");
        assert_eq!(
            (cache.max_entries, cache.max_bytes),
            (GRID_CACHE_ENTRIES, GRID_CACHE_BYTES),
            "a tower is built with the production limits"
        );
    }
    // Grids 1..=9 fit alone (3 and 4 together, 4 and 5 not), 10 and up are
    // over budget.
    let budget = wide_grid_bytes(&tower, 9);
    assert!(wide_grid_bytes(&tower, 10) > budget);
    assert!(wide_grid_bytes(&tower, 3) + wide_grid_bytes(&tower, 4) <= budget);
    assert!(wide_grid_bytes(&tower, 4) + wide_grid_bytes(&tower, 5) > budget);
    *tower.lock_grids().expect("the grid cache lock") =
        GridCache::with_limits(GRID_CACHE_ENTRIES, budget);
    for k in 1..=12 {
        let (keys_before, bytes_before) = {
            let cache = tower.lock_grids().expect("the grid cache lock");
            (cache.keys(), cache.held_bytes)
        };
        let (rows, _) = tower.encode(&wide_image(k), 64).expect("encode");
        let (expected, _) = reference
            .encode(&wide_image(k), 64)
            .expect("the production-limit tower's encode");
        assert_eq!(
            bits(&rows),
            bits(&expected),
            "grid {k}: the rows of the production-limit tower"
        );
        let cache = tower.lock_grids().expect("the grid cache lock");
        assert_cache_invariant(&cache, &format!("after encode {k}"));
        if wide_grid_bytes(&tower, k) <= budget {
            assert_eq!(
                cache.keys().last(),
                Some(&wide_grid(k)),
                "grid {k} fits, so it is cached as the newest"
            );
        } else {
            assert_eq!(
                cache.keys(),
                keys_before,
                "grid {k} is over budget: not cached, nothing evicted"
            );
            assert_eq!(cache.held_bytes, bytes_before, "grid {k}: the total");
        }
    }
    // Grid 9 took the whole budget; the over-budget grids after it left it be.
    assert_eq!(cached_grids(&tower), vec![wide_grid(9)]);
    // An over-budget grid is built again on every request: equal rows, neither
    // cached.
    let (grid_h, grid_w) = wide_grid(11);
    let first = tower.grid_rows(grid_h, grid_w).expect("the rows");
    let second = tower.grid_rows(grid_h, grid_w).expect("the rows again");
    assert!(!Arc::ptr_eq(&first, &second), "built again, not found");
    assert_eq!(bits(&first.pos), bits(&second.pos), "position rows");
    assert_eq!(bits(&first.cos), bits(&second.cos), "cosine rows");
    assert_eq!(bits(&first.sin), bits(&second.sin), "sine rows");
    assert_eq!(first.byte_len(), wide_grid_bytes(&tower, 11));
    assert_eq!(Arc::strong_count(&first), 1, "only this handle holds them");
    assert_eq!(
        cached_grids(&tower),
        vec![wide_grid(9)],
        "still only grid 9"
    );
    // A grid that fits is cached again, making room by evicting grid 9.
    tower.encode(&wide_image(1), 64).expect("encode grid 1");
    assert_eq!(cached_grids(&tower), vec![wide_grid(1)]);
    assert_cache_invariant(&tower.lock_grids().expect("the grid cache lock"), "grid 1");

    // A budget below the smallest grid: nothing is ever cached, and every
    // encode still matches.
    *tower.lock_grids().expect("the grid cache lock") =
        GridCache::with_limits(GRID_CACHE_ENTRIES, wide_grid_bytes(&tower, 1) - 4);
    for k in [1, 2, 3, 1] {
        let (rows, _) = tower.encode(&wide_image(k), 64).expect("encode");
        let (expected, _) = reference
            .encode(&wide_image(k), 64)
            .expect("the production-limit tower's encode");
        assert_eq!(bits(&rows), bits(&expected), "grid {k} on an empty budget");
        let cache = tower.lock_grids().expect("the grid cache lock");
        assert!(cache.keys().is_empty(), "grid {k}: {:?}", cache.keys());
        assert_eq!(cache.held_bytes, 0);
    }
    // The production limits again: the same grid is cached.
    *tower.lock_grids().expect("the grid cache lock") = GridCache::default();
    tower.encode(&wide_image(1), 64).expect("encode grid 1");
    assert_eq!(cached_grids(&tower), vec![wide_grid(1)]);
    assert_eq!(
        tower.lock_grids().expect("the grid cache lock").held_bytes,
        wide_grid_bytes(&tower, 1)
    );
}

/// The byte budget under concurrency: with the cache cut to the rows of the
/// tiny projector's ninth wide grid, encodes on several threads — of grids that
/// fit and of grids too large to be cached at all — evict and refuse grids
/// while other encodes hold their rows, and every encode is bit-identical to
/// the same image on a tower with the production limits; the cache ends within
/// both bounds with its total exact, and holds none of the over-budget grids.
#[test]
fn concurrent_encodes_stay_bit_identical_while_the_byte_budget_evicts_and_refuses_grids() {
    if !metal_available() {
        return;
    }
    let tower = tiny_tower();
    let baseline = tiny_tower();
    let budget = wide_grid_bytes(&tower, 9);
    assert!((10..=CONCURRENT_SHAPES).all(|k| wide_grid_bytes(&tower, k) > budget));
    *tower.lock_grids().expect("the grid cache lock") =
        GridCache::with_limits(GRID_CACHE_ENTRIES, budget);
    let mismatches = concurrent_encode_mismatches(&tower, &baseline);
    assert!(mismatches.is_empty(), "{mismatches:#?}");
    let cache = tower.lock_grids().expect("the grid cache lock");
    assert_cache_invariant(&cache, "after the concurrent encodes");
    for k in 10..=CONCURRENT_SHAPES {
        assert!(
            !cache.keys().contains(&wide_grid(k)),
            "grid {k} is over budget: {:?}",
            cache.keys()
        );
    }
}
