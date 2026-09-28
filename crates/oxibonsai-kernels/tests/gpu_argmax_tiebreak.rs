//! Acceptance test for the GPU `argmax` tie-break rule (`perf-11` / `RT-22`,
//! §5.4 of the legacy CPU-vs-Metal divergence report).
//!
//! The runtime gates every GPU greedy path behind
//! `GPU_ARGMAX_TIEBREAK_IS_FIRST_INDEX` because the shipped MSL kernel used a
//! strict `>` in its tree reduction — a comparison on the **reduction slot**,
//! not on the original index. Each thread strides by
//! `threads_per_threadgroup` (1024), and the surviving maximum then depended
//! on the fold path, so the pre-fix rule was neither global-first nor even
//! minimal in `(i mod 1024, i)`.
//!
//! Measured on an M3 against the unfixed kernel, this file reported:
//! * equal maxima at 1000 and 2000 → **2000** (the report's worked example);
//! * **144 of 320** randomized multi-way ties → a non-minimal index, e.g.
//!   ties at `[358, 418, 513, 529, 997]` in a 1025-wide row → **418**, which
//!   is neither the first index (358) nor the lowest-slot one (358).
//!
//! After the fix all 320 cases return the minimal tied index.
//!
//! This test compiles the **shipped** `MSL_ARGMAX` source (the same string
//! `MetalPipelines::compile` puts in the combined metallib) and dispatches it
//! with the **shipped** geometry (`dispatch_argmax`: one threadgroup, 1024
//! threads), then requires the kernel to return the **minimal** tied index —
//! the `argmax_first` rule the CPU sampler and llama.cpp use.
//!
//! Coverage (the verifier is explicit that the 1000/2000 example alone is not
//! sufficient):
//! * ≥ 300 randomized seeds, tie multiplicity 2..=16;
//! * tie positions placed *across* threadgroup slots (the discriminating
//!   shape), *within* one slot (`i`, `i+1024`, … — this is what pins the
//!   per-thread strided scan), at index 0, and at the last index;
//! * vocab sizes that are not multiples of the threadgroup size, including
//!   Bonsai 2's real 248 320.
//!
//! Off-Metal (or on a host with no Metal device) every test skips honestly
//! with a printed reason rather than passing vacuously.

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use oxibonsai_kernels::gpu_backend::kernel_sources::MSL_ARGMAX;

/// Threads per threadgroup used by the production dispatch
/// (`metal_dispatch.rs::dispatch_argmax`). Not a tunable: the test must
/// exercise the geometry that ships.
const THREADS_PER_GROUP: u64 = 1024;

/// Value planted at every tied maximum. Strictly above the filler range, so
/// the tied set is exactly the planted set and no accidental tie can appear.
const TIE_VALUE: f32 = 42.0;

/// Vocab sizes under test. 248 320 is Bonsai 2's real vocab (and is *not* a
/// multiple of 1024: 242.5 slots); 1025 / 3000 / 100 003 are further
/// non-multiples; 1024 / 2048 / 65 536 are exact multiples.
const VOCABS: &[usize] = &[1024, 1025, 2048, 3000, 4096, 65_536, 100_003, 248_320];

/// Deterministic xorshift64* — failures are reproducible from the seed alone.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Any non-zero state; mixing keeps low seeds from correlating.
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }

    /// Filler value in `[-8.0, 8.0)` — always strictly below [`TIE_VALUE`].
    fn filler(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        unit * 16.0 - 8.0
    }
}

/// CPU oracle: index of the **first** maximal element (`llama.cpp` /
/// `sampling.rs`'s post-`RT-22` rule).
fn argmax_first(data: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut best_idx = 0u32;
    for (i, &v) in data.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Compiled-once Metal harness around the shipped `argmax` kernel.
struct ArgmaxHarness {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    data: metal::Buffer,
    result: metal::Buffer,
    capacity: usize,
}

impl ArgmaxHarness {
    /// `None` when this host has no Metal device (CI runners) — callers skip.
    fn new(capacity: usize) -> Option<Self> {
        let device = Device::system_default()?;
        let library = device
            .new_library_with_source(MSL_ARGMAX, &CompileOptions::new())
            .expect("shipped MSL_ARGMAX must compile");
        let function = library
            .get_function("argmax", None)
            .expect("MSL_ARGMAX must expose `argmax`");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .expect("argmax pipeline creation must succeed");
        assert!(
            pipeline.max_total_threads_per_threadgroup() >= THREADS_PER_GROUP,
            "device rejects the production dispatch geometry: argmax supports only {} threads \
             per threadgroup, dispatch_argmax always asks for {THREADS_PER_GROUP}",
            pipeline.max_total_threads_per_threadgroup()
        );

        // One reusable pair of buffers for every case: 300 × 1 MB of
        // autoreleased allocations is the classic Metal-loop trap.
        let data = device.new_buffer(
            (capacity * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let result = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let queue = device.new_command_queue();
        Some(Self {
            queue,
            pipeline,
            data,
            result,
            capacity,
        })
    }

    /// Run the kernel over `values` with the production dispatch geometry.
    fn argmax(&self, values: &[f32]) -> u32 {
        assert!(
            values.len() <= self.capacity,
            "harness capacity {} exceeded by {}",
            self.capacity,
            values.len()
        );
        metal::objc::rc::autoreleasepool(|| {
            // SAFETY: `StorageModeShared` buffer sized for `capacity` f32s;
            // `values.len() <= capacity` is asserted above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    self.data.contents() as *mut f32,
                    values.len(),
                );
                std::ptr::write(self.result.contents() as *mut u32, u32::MAX);
            }

            let count = values.len() as u32;
            let cmd = self.queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(&self.data), 0);
            encoder.set_buffer(1, Some(&self.result), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                std::ptr::addr_of!(count).cast(),
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(THREADS_PER_GROUP, 1, 1),
            );
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // SAFETY: single u32 written by the kernel into a shared buffer
            // whose completion has been awaited above.
            unsafe { std::ptr::read(self.result.contents() as *const u32) }
        })
    }
}

/// How a case places its tied maxima. Every variant is exercised; only
/// `Spread`/`Random`/`Edges` can distinguish the slot rule from the
/// first-index rule, while `SameSlot` pins the per-thread strided scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TiePlacement {
    /// Uniformly random positions across the whole vocab.
    Random,
    /// Positions deliberately in descending threadgroup slot order, so the
    /// smallest index sits in the *highest* slot — the pre-fix kernel is
    /// maximally wrong here.
    Spread,
    /// Includes index 0 and/or the last index.
    Edges,
    /// All positions congruent mod 1024 (`i`, `i+1024`, …): one single thread
    /// sees every tie, so this pins the per-thread scan's `>` (keep-first).
    SameSlot,
}

impl TiePlacement {
    fn from_index(i: usize) -> Self {
        match i % 4 {
            0 => Self::Random,
            1 => Self::Spread,
            2 => Self::Edges,
            _ => Self::SameSlot,
        }
    }
}

/// Build one case: filler noise plus `tie_count` planted maxima.
/// Returns `(values, tie_positions_sorted)`.
fn build_case(
    rng: &mut Rng,
    vocab: usize,
    tie_count: usize,
    placement: TiePlacement,
) -> (Vec<f32>, Vec<usize>) {
    let mut values: Vec<f32> = (0..vocab).map(|_| rng.filler()).collect();

    let tpg = THREADS_PER_GROUP as usize;
    let mut positions: Vec<usize> = Vec::with_capacity(tie_count);
    match placement {
        TiePlacement::Random => {
            while positions.len() < tie_count {
                let p = rng.below(vocab);
                if !positions.contains(&p) {
                    positions.push(p);
                }
            }
        }
        TiePlacement::Spread => {
            // Slot of the k-th planted index strictly decreases while the
            // index itself strictly increases: index order and slot order
            // are then exactly opposed.
            let slot_step = 1 + rng.below(tpg / (tie_count + 1)).max(1);
            let mut slot = (tie_count * slot_step).min(tpg - 1);
            let mut group = rng.below((vocab / tpg).max(1));
            for _ in 0..tie_count {
                let idx = group * tpg + slot;
                if idx < vocab && !positions.contains(&idx) {
                    positions.push(idx);
                }
                slot = slot.saturating_sub(slot_step);
                group += 1;
                if group * tpg >= vocab {
                    group = 0;
                }
            }
            // Degenerate geometry (tiny vocab) can drop entries; top up.
            while positions.len() < 2 {
                let p = rng.below(vocab);
                if !positions.contains(&p) {
                    positions.push(p);
                }
            }
        }
        TiePlacement::Edges => {
            positions.push(0);
            positions.push(vocab - 1);
            while positions.len() < tie_count {
                let p = rng.below(vocab);
                if !positions.contains(&p) {
                    positions.push(p);
                }
            }
        }
        TiePlacement::SameSlot => {
            let slot = rng.below(tpg.min(vocab));
            let groups = (vocab - slot).div_ceil(tpg);
            let take = tie_count.min(groups).max(1);
            for g in 0..take {
                let idx = slot + g * tpg;
                if idx < vocab {
                    positions.push(idx);
                }
            }
            while positions.len() < 2 {
                let p = rng.below(vocab);
                if !positions.contains(&p) {
                    positions.push(p);
                }
            }
        }
    }

    for &p in &positions {
        values[p] = TIE_VALUE;
    }
    positions.sort_unstable();
    (values, positions)
}

/// The report's worked example, pinned by name: equal maxima at 1000 and 2000
/// in a real Bonsai 2-sized logit row must resolve to **1000**. The pre-fix
/// kernel returns 2000 (`2000 mod 1024 = 976 < 1000`).
#[test]
fn msl_argmax_ties_at_1000_and_2000_resolve_to_the_first_index() {
    let vocab = 248_320usize;
    let Some(harness) = ArgmaxHarness::new(vocab) else {
        eprintln!("SKIPPED msl_argmax_ties_at_1000_and_2000: no Metal device on this host");
        return;
    };

    let mut rng = Rng::new(0xDEAD_BEEF);
    let mut values: Vec<f32> = (0..vocab).map(|_| rng.filler()).collect();
    values[1000] = TIE_VALUE;
    values[2000] = TIE_VALUE;

    let got = harness.argmax(&values);
    assert_eq!(
        argmax_first(&values),
        1000,
        "CPU oracle sanity: the first maximal index must be 1000"
    );
    assert_eq!(
        got, 1000,
        "MSL argmax returned {got} for equal maxima at 1000 and 2000; the first-index rule \
         requires 1000 (slot rule would answer 2000)"
    );
}

/// Randomized acceptance: ≥ 300 planted multi-way ties must all resolve to the
/// minimal tied index. Reports pass/fail counts, then fails with the first
/// offending cases spelled out.
#[test]
fn msl_argmax_returns_the_minimal_tied_index_over_randomized_cases() {
    const CASES: usize = 320;

    let capacity = VOCABS.iter().copied().max().unwrap_or(248_320);
    let Some(harness) = ArgmaxHarness::new(capacity) else {
        eprintln!("SKIPPED msl_argmax_randomized_tiebreak: no Metal device on this host");
        return;
    };

    let mut failures: Vec<String> = Vec::new();
    let mut passed = 0usize;
    // Cases whose tied set spans more than one threadgroup slot: only these
    // can expose a slot/fold-order rule, because ties inside a single slot are
    // resolved by the per-thread scan (which was already first-index correct).
    let mut cross_slot = 0usize;
    let mut placement_counts = [0usize; 4];

    for case in 0..CASES {
        let mut rng = Rng::new(case as u64 + 1);
        let vocab = VOCABS[rng.below(VOCABS.len())];
        let tie_count = 2 + rng.below(15); // 2..=16
        let placement = TiePlacement::from_index(case);
        placement_counts[case % 4] += 1;

        let (values, positions) = build_case(&mut rng, vocab, tie_count, placement);
        let expected = *positions.first().expect("every case plants >= 2 ties");
        let tpg = THREADS_PER_GROUP as usize;
        let slot_winner = positions
            .iter()
            .copied()
            .min_by_key(|&p| (p % tpg, p))
            .unwrap_or(expected);
        if positions.iter().any(|&p| p % tpg != expected % tpg) {
            cross_slot += 1;
        }

        let got = harness.argmax(&values) as usize;
        let oracle = argmax_first(&values) as usize;
        assert_eq!(
            oracle, expected,
            "oracle disagrees with the planted minimum (case {case})"
        );

        if got == expected {
            passed += 1;
        } else {
            failures.push(format!(
                "case {case}: vocab={vocab} ties={:?} placement={placement:?} expected={expected} \
                 got={got} (slot-rule winner would be {slot_winner})",
                positions
            ));
        }
    }

    println!(
        "msl_argmax randomized tie-break: cases={CASES} passed={passed} failed={} \
         cross_slot={cross_slot} (placements: random={} spread={} edges={} same_slot={})",
        failures.len(),
        placement_counts[0],
        placement_counts[1],
        placement_counts[2],
        placement_counts[3]
    );
    assert!(
        cross_slot >= CASES / 2,
        "test would be near-vacuous: only {cross_slot}/{CASES} cases plant ties in more than \
         one threadgroup slot, and same-slot ties cannot expose a fold-order tie rule"
    );

    let shown: Vec<&String> = failures.iter().take(8).collect();
    assert!(
        failures.is_empty(),
        "{} / {CASES} randomized tie cases returned a non-minimal index; first failures:\n{}",
        failures.len(),
        shown
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Non-tied sanity: a unique maximum must be found regardless of where it sits
/// (guards against the tie-break edit breaking the ordinary path), including a
/// vocab shorter than one threadgroup, where most threads scan nothing.
#[test]
fn msl_argmax_finds_a_unique_maximum_anywhere() {
    let capacity = 248_320usize;
    let Some(harness) = ArgmaxHarness::new(capacity) else {
        eprintln!("SKIPPED msl_argmax_unique_maximum: no Metal device on this host");
        return;
    };

    let mut checked = 0usize;
    for (case, &vocab) in VOCABS.iter().enumerate() {
        for &pos in &[0usize, 1, 1023, 1024, vocab / 2, vocab - 1] {
            if pos >= vocab {
                continue;
            }
            let mut rng = Rng::new((case * 977 + pos) as u64 + 1);
            let mut values: Vec<f32> = (0..vocab).map(|_| rng.filler()).collect();
            values[pos] = TIE_VALUE;
            let got = harness.argmax(&values) as usize;
            assert_eq!(
                got, pos,
                "unique maximum at {pos} of vocab {vocab} was reported as {got}"
            );
            checked += 1;
        }
    }

    // Short rows (fewer elements than threads): idle threads must not win.
    for vocab in [1usize, 2, 7, 33, 512, 1023] {
        let mut rng = Rng::new(vocab as u64 * 31 + 7);
        let mut values: Vec<f32> = (0..vocab).map(|_| rng.filler()).collect();
        let pos = vocab - 1;
        values[pos] = TIE_VALUE;
        let got = harness.argmax(&values) as usize;
        assert_eq!(
            got, pos,
            "short row (vocab {vocab}): maximum at {pos} reported as {got}"
        );
        checked += 1;
    }

    println!("msl_argmax unique-maximum sanity: {checked} placements checked");
}
