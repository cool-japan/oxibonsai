//! ACCEPTANCE (KERN-PARALLEL / `K-15`): **no heap allocation inside any GEMV
//! call path** once the thread-local scratch is warm.
//!
//! This is the counting-allocator test the parallel-GEMV package's acceptance
//! criterion asked for. It lives in its own integration binary on purpose: a
//! process-wide `#[global_allocator]` installed in the crate's `src/` test
//! module would contaminate every other unit test sharing that binary, which
//! is why the in-crate substitute
//! (`parallel_tests::kquant_driver_repeated_calls_do_not_regress_to_per_call_alloc_timing`)
//! could only be a wall-clock proxy — and is `#[ignore]`d, so it never ran in
//! a gate. This test is **not** `#[ignore]`d.
//!
//! # Why the first version of this file was ~50% flaky
//!
//! `System`, wrapped in one **process-wide** `AtomicUsize`, cannot tell "the
//! call under test allocated" from "some other thread allocated while the
//! call under test happened to be running". Every `_par` path hands its real
//! work to Rayon's global thread pool, and that pool's own infrastructure
//! (crossbeam-deque `Worker`/`Injector` buffer growth, amortized roughly
//! every 32 pushed jobs, plus assorted per-worker bookkeeping) allocates
//! **asynchronously**: a worker thread can still be draining or growing a
//! queue from a *previous* dispatch while the *next* call's measurement
//! window opens on the main thread, so the stray allocation lands wherever
//! the wall clock happens to put it. A single warm-up pass cannot drain this
//! — it is not a one-time start-up cost, it recurs on a schedule tied to how
//! many jobs have been pushed overall, not to whether any particular
//! thread-local scratch buffer is warm. The proof it was exactly this: some
//! runs flagged the **sequential** shape (`n_rows = 4`, which never touches
//! Rayon at all), which is only possible if the flagged allocation was made
//! by a *different* thread and landed, by pure timing, inside whichever
//! window happened to be open on the main thread when it occurred.
//!
//! # The fix: two independent, thread-aware axes
//!
//! 1. **Thread-local, exact-zero (axis 1).** Every allocation/reallocation
//!    bumps a `thread_local!` counter on whichever thread made it.
//!    [`measure`] reads only the *calling* thread's own counters before and
//!    after `f()` runs, so Rayon worker-thread activity — living in a
//!    different thread's `thread_local!` cell entirely — can never land in
//!    this window, regardless of when it happens. This is fully
//!    deterministic. For the `[sequential]` shape, which never leaves the
//!    calling thread, it is the whole story and exactly closes the flakiness
//!    above. For the `[parallel]` shape it still catches anything the entry
//!    point itself allocates before handing off to Rayon.
//! 2. **Process-wide, size-gated (axis 2).** A `_par` path's real
//!    row-processing work runs *on Rayon's worker threads*, invisible to
//!    axis 1 by construction, so a genuine worker-side regression (the
//!    historical `vec![0.0f32; in_features]`) needs a second, thread-agnostic
//!    net: any single allocation at or above [`LARGE_ALLOC_THRESHOLD_BYTES`]
//!    bumps a process-wide counter instead of the per-thread one, while
//!    [`K`] (this file's shared inner dimension) is sized so that a
//!    reintroduced `vec![0.0f32; in_features]` allocates
//!    `K * size_of::<f32>()` bytes — see the constant's own doc for exactly
//!    what margin this buys and, as important, what it does not promise.
//!    Axis 2 is honest about what it is: a narrowing, not a structural fix
//!    like axis 1. It is still process-wide and still timing-dependent in
//!    principle — a same-window misattribution of a Rayon-internal
//!    allocation is not made *impossible*, only made to require that
//!    allocation be at least [`LARGE_ALLOC_THRESHOLD_BYTES`] bytes, which is
//!    a much rarer event than the pre-fix counter's "any allocation of any
//!    size, on any thread" trigger. Measured on this host (below) that
//!    rarer event has a rate of zero across 40 runs even with the threshold
//!    lowered to `1` (i.e. with the narrowing removed entirely) — evidence
//!    that the specific noise observed does not reach the sizes
//!    axis 2 would need to see to misfire, not a proof that no environment
//!    ever could produce it.
//!
//! Neither axis is "a small nonzero allocation budget": both still assert
//! **exactly zero** of what they count. Axis 2 simply narrows what counts as
//! an allocation of concern to sizes no legitimate Rayon bookkeeping call
//! reaches, rather than tolerating a bounded number of otherwise-unfiltered
//! allocations. [`assert_instrument_is_armed`] proves both axes actually
//! detect an allocation — including one made inside a Rayon-dispatched
//! closure, the harder case — before the real measurements below are
//! trusted; a silently broken counter would make every "zero" downstream
//! meaningless.
//!
//! One precondition is made deterministic rather than left to scheduling:
//! [`warm_kquant_scratch_on_every_rayon_thread`] warms the K-quant drivers'
//! per-thread scratch row on *every* Rayon pool thread before anything is
//! measured, because a single warm-up call per path cannot guarantee that
//! every worker took part in one (that function's doc gives the failure rate
//! it closed).
//!
//! Axis 2 is still process-wide, so this file deliberately contains a single
//! `#[test]`: a concurrently running test in the same binary could trip the
//! size guard with a large allocation of its own. (Each `tests/*.rs` file is
//! its own binary, so this only constrains this file, not the crate's other
//! integration tests.)
//!
//! Both shapes of every path are measured:
//! * `[parallel]` — `n_rows` above the platform-tuned thresholds, i.e. the
//!   Rayon branch production decode actually takes for an LM head;
//! * `[sequential]` — `n_rows = 4`, the short-circuit branch, which is where
//!   `K-15`'s per-call `vec![0.0f32; in_features]` used to live.
//!
//! Tier: the asserted pass runs on `KernelDispatcher::with_tier(cpu_kernel_tier())`,
//! because the zero-allocation property is a property of the **CPU** GEMV
//! kernels — a call that actually reaches a Metal kernel allocates a command
//! buffer and encoder by construction, which is a property of the encoder and
//! not of the GEMV arithmetic. The same measurement is therefore *also* run on
//! `KernelDispatcher::auto_detect()` (the tier this host really resolves, `Gpu`
//! here) and **reported, not asserted** — see the printed output for this
//! fixture's actual numbers on that tier.
//!
//! Coverage limit worth naming: on AArch64 `gemv_q4k`/`q6k`/`q8k` route to
//! `gemv_kquant_row_parallel_fused`, which materialises no row buffer at all
//! and is therefore zero-allocation *by construction* rather than by scratch
//! reuse. The thread-local `KQUANT_ROW_SCRATCH` path this test was written for
//! (`gemv_kquant_row_parallel`) is the non-AArch64 branch, and the driver is
//! `pub(crate)`, so an integration test cannot call it directly on this host.
//!
//! ## `OXIBONSAI_KERNEL_TIER` (K-14)
//!
//! The opt-in INT8 dot-product tier quantizes its activation once per
//! GEMV/GEMM call, an allocation this file's whole premise asserts is zero.
//! [`TierEnvGuard::cleared`] therefore clears the variable for this
//! process's one test, so an ambient shell export cannot turn the
//! zero-allocation assertion into a false failure, and the gate proves the
//! guard actually works by exporting `OXIBONSAI_KERNEL_TIER=neon-dot` around
//! this binary anyway.

#![cfg(not(target_arch = "wasm32"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use half::f16;
use oxibonsai_core::{
    BlockFP8E4M3, BlockFP8E5M2, BlockQ1_0G128, BlockQ4K, BlockQ4_0, BlockQ6K, BlockQ8K, BlockQ8_0,
    BlockTQ2_0_g128,
};
use oxibonsai_kernels::dispatch_int8::KERNEL_TIER_ENV;
use oxibonsai_kernels::{cpu_kernel_tier, KernelDispatcher, PlatformProfile};

/// Serializes this binary's own access to `OXIBONSAI_KERNEL_TIER`; mirrors
/// `cross_backend_determinism_tests.rs`'s `TierEnvGuard` /
/// `int8_tier_parity.rs`'s `EnvGuard` in the same crate. This file has a
/// single `#[test]`, so nothing else in-process can race it today, but the
/// guard is still process-wide (not merely a local snapshot) so it stays
/// correct if a second test is ever added here.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII owner of `OXIBONSAI_KERNEL_TIER` for one test: takes [`ENV_LOCK`],
/// snapshots and clears the variable, and restores the snapshot on drop —
/// also while unwinding from a failed assertion.
struct TierEnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prior: Option<String>,
}

impl TierEnvGuard {
    fn cleared() -> Self {
        let lock = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var(KERNEL_TIER_ENV).ok();
        // SAFETY: `lock` is held for the lifetime of the returned guard and
        // serializes every reader/writer of the variable in this binary.
        unsafe {
            std::env::remove_var(KERNEL_TIER_ENV);
        }
        Self { _lock: lock, prior }
    }
}

impl Drop for TierEnvGuard {
    fn drop(&mut self) {
        // SAFETY: `self._lock` is held for the entire body of `drop`.
        unsafe {
            match &self.prior {
                Some(v) => std::env::set_var(KERNEL_TIER_ENV, v),
                None => std::env::remove_var(KERNEL_TIER_ENV),
            }
        }
    }
}

// ── counting global allocator ───────────────────────────────────────────────

thread_local! {
    /// Axis 1: allocations/reallocations attributed to *this* thread only.
    /// [`measure`] reads these with plain `.with()` from the one thread that
    /// will ever consume the value; the allocator hooks below use
    /// `try_with` instead because they run on every thread in the process,
    /// including one that may be mid-teardown when a `dealloc`-adjacent
    /// allocation occurs (`.with()` panics in that window, `try_with` does
    /// not).
    static THREAD_ALLOCS: Cell<usize> = const { Cell::new(0) };
    static THREAD_REALLOCS: Cell<usize> = const { Cell::new(0) };
}

/// Axis 2's threshold — sized as a **floor below the bug, not a ceiling above
/// the noise**. Any single allocation at or above this many bytes bumps the
/// process-wide counter. This deliberately does *not* claim to sit above
/// Rayon's own work-stealing bookkeeping (crossbeam-deque `Worker`/`Injector`
/// buffers grow geometrically from a small base, so a run with enough
/// pushed jobs could plausibly produce an internal allocation *at* a
/// power-of-two near this value — there is no principled ceiling to reason
/// from here, only an empirical one). What this constant actually needs is
/// to sit **below** [`K`]'s buggy-allocation size (`K * size_of::<f32>()` =
/// 32 KiB) with enough margin that the two are never confusable: `4096` is
/// an 8x margin under that, a round number, and nothing more load-bearing
/// than that.
///
/// Measured, not just reasoned about: with this constant temporarily
/// lowered to `1` (axis 2 then counts *every* allocation of any size,
/// anywhere in the process — the same blast radius this file's pre-fix,
/// purely process-wide counter had), 40 repetitions of the compiled binary
/// on this host recorded **zero** allocations in any `warm_then_measure`
/// window, including the historically-flaky `[sequential]` shapes. On this
/// host and build the background-noise floor is not merely small, it is
/// absent — but that is a property of this environment, not a guarantee
/// this constant enforces. A future host, allocator, or Rayon version that
/// does produce a background allocation at or above 4096 bytes will fail
/// this test loudly, naming the offending path and byte count (see the
/// assertion message below) rather than flaking silently — exactly the
/// "report the count and the path rather than relaxing the assertion"
/// response the acceptance criterion asks for, not a defect in the
/// threshold's choice.
const LARGE_ALLOC_THRESHOLD_BYTES: usize = 4096;

/// Process-wide, size-gated allocation count (axis 2).
static LARGE_ALLOCS: AtomicUsize = AtomicUsize::new(0);
/// Largest single allocation observed since the last reset. Diagnostics
/// only — the pass/fail signal is [`LARGE_ALLOCS`]'s delta, not this value.
static LARGE_ALLOC_MAX_BYTES: AtomicUsize = AtomicUsize::new(0);

/// `System`, plus a per-thread counter and a size-gated process-wide counter
/// on every allocating entry point. See the module doc for why both exist.
struct CountingAlloc;

// SAFETY: every method forwards to `System` unchanged; the counters are
// side-effect-free (`try_with` / relaxed atomics) and never themselves
// allocate, so recursion through this allocator cannot occur.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_alloc(layout.size());
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_alloc(layout.size());
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // The bigger of the old and new sizes is what matters for "is this a
        // problem-sized buffer" — a shrink should not hide a large buffer,
        // and a growth's interesting size is the one being grown *to*.
        record_realloc(layout.size().max(new_size));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Record one allocation: bump the *calling thread's own* counter (best
/// effort, see [`THREAD_ALLOCS`]'s doc for why this is `try_with`), and, when
/// it is large enough to only plausibly be a genuine row buffer, the
/// process-wide guard too.
#[inline]
fn record_alloc(size: usize) {
    let _ = THREAD_ALLOCS.try_with(|c| c.set(c.get() + 1));
    note_if_large(size);
}

#[inline]
fn record_realloc(size: usize) {
    let _ = THREAD_REALLOCS.try_with(|c| c.set(c.get() + 1));
    note_if_large(size);
}

#[inline]
fn note_if_large(size: usize) {
    if size >= LARGE_ALLOC_THRESHOLD_BYTES {
        LARGE_ALLOCS.fetch_add(1, Ordering::Relaxed);
        LARGE_ALLOC_MAX_BYTES.fetch_max(size, Ordering::Relaxed);
    }
}

/// Zero-allocation evidence for one measured call. See the module doc for
/// why both axes are needed: axis 1 alone would miss a worker-thread
/// regression, axis 2 alone would miss a small allocation the entry point
/// makes directly on the calling thread.
#[derive(Debug, Clone, Copy)]
struct Measurement {
    /// Axis 1: allocations/reallocations attributed to the thread that made
    /// this call.
    thread_allocs: usize,
    thread_reallocs: usize,
    /// Axis 2: allocations of size >= [`LARGE_ALLOC_THRESHOLD_BYTES`],
    /// anywhere in the process, observed during the call.
    large_allocs: usize,
    /// Diagnostics only: the biggest single allocation observed while this
    /// call ran; `0` when `large_allocs == 0`.
    large_alloc_max_bytes: usize,
}

impl Measurement {
    fn is_clean(&self) -> bool {
        self.thread_allocs == 0 && self.thread_reallocs == 0 && self.large_allocs == 0
    }
}

/// Run `f` on the calling thread and report what both axes saw during it.
fn measure(f: impl FnOnce()) -> Measurement {
    let a0 = THREAD_ALLOCS.with(Cell::get);
    let r0 = THREAD_REALLOCS.with(Cell::get);
    let l0 = LARGE_ALLOCS.load(Ordering::Relaxed);
    LARGE_ALLOC_MAX_BYTES.store(0, Ordering::Relaxed);
    f();
    Measurement {
        thread_allocs: THREAD_ALLOCS.with(Cell::get) - a0,
        thread_reallocs: THREAD_REALLOCS.with(Cell::get) - r0,
        large_allocs: LARGE_ALLOCS.load(Ordering::Relaxed) - l0,
        large_alloc_max_bytes: LARGE_ALLOC_MAX_BYTES.load(Ordering::Relaxed),
    }
}

/// Prove the instrument can see an allocation before trusting it to report
/// zero — a silently broken counter (e.g. a `vec!` the optimizer proved dead
/// and elided) would make every downstream "zero allocations" meaningless.
/// Two axes, independently:
/// * a [`K`]-element `Vec` allocated directly on the calling thread must trip
///   *both* the thread-local counter and the size guard;
/// * the same allocation, made inside a Rayon-dispatched closure (so it may
///   run on a worker thread instead of the calling thread — the exact shape
///   of a genuine `_par` row-buffer regression), must still trip the size
///   guard even though axis 1 cannot see across threads.
///
/// `black_box` keeps the compiler from proving the `Vec` is dead and eliding
/// the allocation entirely.
fn assert_instrument_is_armed() {
    let bytes = K * std::mem::size_of::<f32>();

    let direct = measure(|| {
        let v = vec![0.0f32; K];
        std::hint::black_box(&v);
    });
    assert!(
        direct.thread_allocs >= 1,
        "instrumentation fault: a {bytes}-byte Vec allocated on the calling thread was not seen \
         by the thread-local allocation counter (axis 1 is broken)"
    );
    assert!(
        direct.large_allocs >= 1,
        "instrumentation fault: a {bytes}-byte Vec did not trip the large-allocation guard \
         (threshold {LARGE_ALLOC_THRESHOLD_BYTES} bytes) -- axis 2 is broken"
    );

    // `oper_b` (the second closure) is the one Rayon may hand to an idle
    // worker thread instead of running inline -- see `rayon::join`'s own
    // docs ("operation B is offered to any idle threads nearby"). Whether it
    // actually migrates is a scheduling detail this assertion does not
    // depend on: axis 2 is thread-agnostic by design, so it must see the
    // allocation either way.
    let via_rayon = measure(|| {
        rayon::join(
            || {},
            || {
                let v = vec![0.0f32; K];
                std::hint::black_box(&v);
            },
        );
    });
    assert!(
        via_rayon.large_allocs >= 1,
        "instrumentation fault: a {bytes}-byte Vec allocated inside a Rayon-dispatched closure \
         did not trip the process-wide large-allocation guard -- this is the axis that must \
         catch a genuine worker-side K-15 regression, and it saw nothing"
    );

    println!(
        "gemv_no_alloc: instrumentation self-check OK (direct thread_allocs={}, \
         direct large_allocs={}, via-rayon large_allocs={})",
        direct.thread_allocs, direct.large_allocs, via_rayon.large_allocs
    );
}

// ── deterministic fixtures ──────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    fn byte(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }

    fn signed(&mut self, scale: f32) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale
    }
}

fn input_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    (0..len).map(|_| rng.signed(1.5)).collect()
}

fn raw_weights(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    (0..len).map(|_| rng.signed(0.75)).collect()
}

fn q1_blocks(count: usize, seed: u64) -> Vec<BlockQ1_0G128> {
    let mut rng = Rng(seed | 1);
    (0..count)
        .map(|_| {
            let mut qs = [0u8; 16];
            for q in qs.iter_mut() {
                *q = rng.byte();
            }
            BlockQ1_0G128 {
                d: f16::from_f32(0.25 + rng.signed(0.1)),
                qs,
            }
        })
        .collect()
}

fn tq2_blocks(count: usize, seed: u64) -> Vec<BlockTQ2_0_g128> {
    let mut rng = Rng(seed | 1);
    (0..count)
        .map(|_| {
            let mut qs = [0u8; 32];
            for q in qs.iter_mut() {
                *q = rng.byte();
            }
            BlockTQ2_0_g128 {
                qs,
                d: f16::from_f32(0.3 + rng.signed(0.1)),
            }
        })
        .collect()
}

fn q4k_blocks(count: usize, seed: u64) -> Vec<BlockQ4K> {
    let mut rng = Rng(seed | 1);
    (0..count)
        .map(|_| {
            let mut qs = [0u8; 128];
            for q in qs.iter_mut() {
                *q = rng.byte();
            }
            let mut scales = [0u8; 12];
            for s in scales.iter_mut() {
                *s = rng.byte();
            }
            BlockQ4K {
                d: f16::from_f32(rng.signed(0.05)),
                dmin: f16::from_f32(rng.signed(0.05)),
                scales,
                qs,
            }
        })
        .collect()
}

fn q6k_blocks(count: usize, seed: u64) -> Vec<BlockQ6K> {
    let mut rng = Rng(seed | 1);
    (0..count)
        .map(|_| {
            let mut ql = [0u8; 128];
            for q in ql.iter_mut() {
                *q = rng.byte();
            }
            let mut qh = [0u8; 64];
            for q in qh.iter_mut() {
                *q = rng.byte();
            }
            let mut scales = [0i8; 16];
            for s in scales.iter_mut() {
                *s = rng.byte() as i8;
            }
            BlockQ6K {
                ql,
                qh,
                scales,
                d: f16::from_f32(rng.signed(0.02)),
            }
        })
        .collect()
}

fn q8k_blocks(count: usize, seed: u64) -> Vec<BlockQ8K> {
    let mut rng = Rng(seed | 1);
    (0..count)
        .map(|_| {
            let mut qs = [0i8; 256];
            for q in qs.iter_mut() {
                *q = rng.byte() as i8;
            }
            BlockQ8K {
                d: rng.signed(0.01),
                qs,
                bsums: [0i16; 16],
            }
        })
        .collect()
}

/// One measurable GEMV invocation, pre-allocated (fixtures + output buffer +
/// the boxed closure itself are all built before any snapshot is taken).
type Path = (String, Box<dyn FnMut()>);

/// Inner dimension shared by every path: a multiple of 32 (Q4_0/Q8_0/FP8),
/// 128 (Q1_0_G128/TQ2_0_G128) and 256 (the K-quant super-block) — 8192
/// satisfies all three with plenty of room. Deliberately large: this is what
/// sizes the historical `K-15` buggy allocation (`vec![0.0f32; in_features]`
/// = `K * size_of::<f32>()` = 32 KiB), which must sit unmistakably above
/// [`LARGE_ALLOC_THRESHOLD_BYTES`] for axis 2 to mean anything (see that
/// constant's doc and the module doc's "two independent, thread-aware axes"
/// section). Only the inner (column) dimension — row count is a separate
/// knob below and does not need to be large for this purpose.
const K: usize = 8192;

/// Build every GEMV path at one row count, against one dispatcher.
///
/// `with_kquant` adds the three K-quant entry points, which take no
/// dispatcher (they are tier-independent free functions), so they are built
/// only once rather than duplicated per tier.
fn build_paths(
    dispatcher: &Arc<KernelDispatcher>,
    n_rows: usize,
    shape: &str,
    with_kquant: bool,
) -> Vec<Path> {
    let mut paths: Vec<Path> = Vec::new();
    let input = input_vec(K, 0xA11C);

    // 1-bit Q1_0_G128.
    {
        let blocks = q1_blocks(n_rows * (K / 128), 0x1B17);
        let input = input.clone();
        let mut out = vec![0.0f32; n_rows];
        let d = Arc::clone(dispatcher);
        paths.push((
            format!("gemv_1bit_g128_par [{shape}]"),
            Box::new(move || {
                oxibonsai_kernels::parallel::gemv_1bit_g128_par(
                    &d, &blocks, &input, &mut out, n_rows, K,
                )
                .expect("gemv_1bit_g128_par must succeed");
            }),
        ));
    }

    // Ternary TQ2_0_G128, flat row-parallel.
    {
        let blocks = tq2_blocks(n_rows * (K / 128), 0x7E27);
        let input = input.clone();
        let mut out = vec![0.0f32; n_rows];
        let d = Arc::clone(dispatcher);
        paths.push((
            format!("gemv_ternary_g128_par [{shape}]"),
            Box::new(move || {
                oxibonsai_kernels::parallel::gemv_ternary_g128_par(
                    &d, &blocks, &input, &mut out, n_rows, K,
                )
                .expect("gemv_ternary_g128_par must succeed");
            }),
        ));
    }

    // Ternary TQ2_0_G128, cache-tiled (the K-M2 routing target).
    {
        let blocks = tq2_blocks(n_rows * (K / 128), 0x7117);
        let input = input.clone();
        let mut out = vec![0.0f32; n_rows];
        let d = Arc::clone(dispatcher);
        paths.push((
            format!("gemv_parallel_tiled_ternary [{shape}]"),
            Box::new(move || {
                oxibonsai_kernels::parallel_tiled::gemv_parallel_tiled_ternary(
                    &d, &blocks, &input, &mut out, n_rows, K,
                )
                .expect("gemv_parallel_tiled_ternary must succeed");
            }),
        ));
    }

    // FP8 E4M3 / E5M2.
    {
        let raw = raw_weights(n_rows * K, 0xF8E4);
        let blocks = BlockFP8E4M3::quantize(&raw).expect("quantize fp8 e4m3");
        let input = input.clone();
        let mut out = vec![0.0f32; n_rows];
        let d = Arc::clone(dispatcher);
        paths.push((
            format!("gemv_fp8_e4m3_par [{shape}]"),
            Box::new(move || {
                oxibonsai_kernels::gemv_fp8_e4m3_par(&d, &blocks, &input, &mut out, n_rows, K)
                    .expect("gemv_fp8_e4m3_par must succeed");
            }),
        ));
    }
    {
        let raw = raw_weights(n_rows * K, 0xF8E5);
        let blocks = BlockFP8E5M2::quantize(&raw).expect("quantize fp8 e5m2");
        let input = input.clone();
        let mut out = vec![0.0f32; n_rows];
        let d = Arc::clone(dispatcher);
        paths.push((
            format!("gemv_fp8_e5m2_par [{shape}]"),
            Box::new(move || {
                oxibonsai_kernels::gemv_fp8_e5m2_par(&d, &blocks, &input, &mut out, n_rows, K)
                    .expect("gemv_fp8_e5m2_par must succeed");
            }),
        ));
    }

    // Standard GGUF quants Q4_0 / Q8_0.
    {
        let raw = raw_weights(n_rows * K, 0x0401);
        let blocks = BlockQ4_0::quantize(&raw).expect("quantize q4_0");
        let input = input.clone();
        let mut out = vec![0.0f32; n_rows];
        let d = Arc::clone(dispatcher);
        paths.push((
            format!("gemv_q4_0_par [{shape}]"),
            Box::new(move || {
                oxibonsai_kernels::gemv_q4_0_par(&d, &blocks, &input, &mut out, n_rows, K)
                    .expect("gemv_q4_0_par must succeed");
            }),
        ));
    }
    {
        let raw = raw_weights(n_rows * K, 0x0801);
        let blocks = BlockQ8_0::quantize(&raw).expect("quantize q8_0");
        let input = input.clone();
        let mut out = vec![0.0f32; n_rows];
        let d = Arc::clone(dispatcher);
        paths.push((
            format!("gemv_q8_0_par [{shape}]"),
            Box::new(move || {
                oxibonsai_kernels::gemv_q8_0_par(&d, &blocks, &input, &mut out, n_rows, K)
                    .expect("gemv_q8_0_par must succeed");
            }),
        ));
    }

    if with_kquant {
        // Each Q4_K/Q6_K/Q8_K block is one super-block covering 256 elements,
        // so a row of `K` columns needs `K / 256` of them (this was `1` and
        // implicit back when `K` itself was 256).
        let kquant_blocks = n_rows * (K / 256);
        {
            let blocks = q4k_blocks(kquant_blocks, 0x4B4B);
            let input = input.clone();
            let mut out = vec![0.0f32; n_rows];
            paths.push((
                format!("gemv_q4k [{shape}]"),
                Box::new(move || {
                    oxibonsai_kernels::gemv_q4k(&blocks, &input, &mut out, n_rows, K)
                        .expect("gemv_q4k must succeed");
                }),
            ));
        }
        {
            let blocks = q6k_blocks(kquant_blocks, 0x6B6B);
            let input = input.clone();
            let mut out = vec![0.0f32; n_rows];
            paths.push((
                format!("gemv_q6k [{shape}]"),
                Box::new(move || {
                    oxibonsai_kernels::gemv_q6k(&blocks, &input, &mut out, n_rows, K)
                        .expect("gemv_q6k must succeed");
                }),
            ));
        }
        {
            let blocks = q8k_blocks(kquant_blocks, 0x8B8B);
            let input = input.clone();
            let mut out = vec![0.0f32; n_rows];
            paths.push((
                format!("gemv_q8k [{shape}]"),
                Box::new(move || {
                    oxibonsai_kernels::gemv_q8k(&blocks, &input, &mut out, n_rows, K)
                        .expect("gemv_q8k must succeed");
                }),
            ));
        }
    }

    paths
}

/// Warm every Rayon pool thread's K-quant row scratch before anything is
/// measured.
///
/// `gemv_q4k` / `gemv_q6k` / `gemv_q8k` (on every non-AArch64 target) borrow
/// a per-OS-thread scratch row — `parallel.rs`'s `KQUANT_ROW_SCRATCH` — that
/// grows to `in_features` the first time a thread uses it. One warm-up call
/// per path ([`warm_then_measure`]) cannot warm it on every pool thread:
/// which threads a `[parallel]` call's rows land on is decided by Rayon's
/// work stealing, so a worker that sat out every K-quant warm-up call first
/// touches its scratch inside a *measured* call — a one-time `K * 4` =
/// 32 KiB growth that axis 2 rightly reports. On an 8-core x86-64 host it
/// showed up in 12 of 40 runs of the tree as it stood — `gemv_q4k
/// [parallel]` in 8 of them, `gemv_q6k` in 3, `gemv_q8k` in 2 (one run had
/// two) — on top of the FP8 offenders all 40 of those runs also had, and it
/// still failed 1 run in 40 once only the FP8 gate was fixed. With this
/// warm-up as well, 60 of 60 runs passed, then 80 of 80 with several running
/// concurrently.
///
/// [`rayon::broadcast`] runs one sequential-shape call of each K-quant GEMV
/// on *every* pool thread, so the scratch is warm wherever the measured rows
/// land. That makes this file's premise — "once the thread-local scratch is
/// warm" — deterministic rather than a matter of scheduling luck. It is not
/// an allocation budget: a per-call allocation (the historical
/// `vec![0.0f32; in_features]`, or one `Vec` per Rayon split) still happens
/// on every measured call, warm or not, and still fails both axes.
fn warm_kquant_scratch_on_every_rayon_thread() {
    // One row is below every tuned threshold, so each call takes the
    // sequential branch on whichever pool thread `broadcast` runs it, which
    // grows *that* thread's scratch to `K`.
    const ROWS: usize = 1;
    let input = input_vec(K, 0xA11C);
    let q4k = q4k_blocks(ROWS * (K / 256), 0x4B4B);
    let q6k = q6k_blocks(ROWS * (K / 256), 0x6B6B);
    let q8k = q8k_blocks(ROWS * (K / 256), 0x8B8B);
    rayon::broadcast(|_| {
        let mut out = [0.0f32; ROWS];
        oxibonsai_kernels::gemv_q4k(&q4k, &input, &mut out, ROWS, K)
            .expect("warm-up gemv_q4k must succeed");
        oxibonsai_kernels::gemv_q6k(&q6k, &input, &mut out, ROWS, K)
            .expect("warm-up gemv_q6k must succeed");
        oxibonsai_kernels::gemv_q8k(&q8k, &input, &mut out, ROWS, K)
            .expect("warm-up gemv_q8k must succeed");
    });
}

/// Warm every path once, then measure the second call of each.
fn warm_then_measure(paths: &mut [Path]) -> Vec<(String, Measurement)> {
    for (_, call) in paths.iter_mut() {
        call();
    }
    let mut results = Vec::with_capacity(paths.len());
    for (name, call) in paths.iter_mut() {
        let measurement = measure(call);
        results.push((name.clone(), measurement));
    }
    results
}

#[test]
fn gemv_call_paths_do_not_allocate_after_warmup() {
    // Clear the tier selector for this test's whole run — see the
    // module doc's `OXIBONSAI_KERNEL_TIER` section. Held for the entire
    // test, not just the measured windows, so a concurrent mutation from
    // elsewhere in this process (there is none today; see `ENV_LOCK`'s doc)
    // could never straddle a `warm_then_measure` call.
    let _env = TierEnvGuard::cleared();

    assert_instrument_is_armed();

    let thresholds = PlatformProfile::global_thresholds();
    // Above every tuned threshold => the Rayon branch of each entry point
    // (including `gemv_parallel_tiled_ternary`'s L2-tile split) is the branch
    // under measurement, not the sequential short-circuit. This only needs
    // to CLEAR the thresholds, not be huge: the size of the buggy allocation
    // this test is designed to catch comes from `K` (the column count), not
    // from the row count, so there is no reason to grow this beyond 600.
    let parallel_rows = 600
        .max(thresholds.par_gemv_min_rows + 1)
        .max(thresholds.par_tiled_min_rows + 1)
        .min(8192);
    // T-09 fixture precondition (same failure mode `parallel_tests.rs`'s six
    // `*_par_byte_identical_to_per_row_dispatch` guards exist for): the
    // `.min(8192)` cap above exists only to keep fixture construction cheap
    // on a platform with an unusually low tuned threshold, but on a platform
    // where `par_tiled_min_rows` is itself >= 8192, that cap would silently
    // win over the `.max(par_tiled_min_rows + 1)` just before it, and
    // `gemv_parallel_tiled_ternary` would then take its non-tiled branch
    // instead of the one this test claims to measure -- passing even with
    // the tiled path's own K-15 fix deleted. Assert the ordering explicitly
    // rather than let the cap's silent precedence hide that.
    assert!(
        parallel_rows > thresholds.par_tiled_min_rows,
        "fixture assumption: parallel_rows={parallel_rows} must clear par_tiled_min_rows ({}); \
         the 8192 cap has overridden the tuned threshold on this host, so \
         gemv_parallel_tiled_ternary would silently test its non-tiled branch",
        thresholds.par_tiled_min_rows
    );
    // Below every tuned threshold => the sequential fallback, which is where
    // K-15's per-call `vec![0.0f32; in_features]` used to be allocated.
    let sequential_rows = 4usize;
    assert!(
        sequential_rows < thresholds.par_gemv_min_rows,
        "fixture assumption: n_rows={sequential_rows} must be below par_gemv_min_rows ({})",
        thresholds.par_gemv_min_rows
    );

    // Per-thread scratch must be warm on every pool thread, not only on the
    // ones `warm_then_measure`'s single warm-up call happened to use.
    warm_kquant_scratch_on_every_rayon_thread();

    let cpu = Arc::new(KernelDispatcher::with_tier(cpu_kernel_tier()));
    let auto = Arc::new(KernelDispatcher::auto_detect());
    let cpu_tier = cpu.tier();
    let auto_tier = auto.tier();

    // Everything below is built (and therefore allocated) before any snapshot.
    let mut cpu_paths = build_paths(&cpu, parallel_rows, "parallel", true);
    cpu_paths.extend(build_paths(&cpu, sequential_rows, "sequential", true));
    let mut auto_paths = build_paths(&auto, parallel_rows, "parallel", false);
    auto_paths.extend(build_paths(&auto, sequential_rows, "sequential", false));

    let cpu_results = warm_then_measure(&mut cpu_paths);
    let auto_results = warm_then_measure(&mut auto_paths);

    // Printing allocates, so it happens strictly outside the measured windows.
    println!(
        "gemv_no_alloc: k={K} parallel_rows={parallel_rows} sequential_rows={sequential_rows} \
         par_gemv_min_rows={} par_tiled_min_rows={} large_alloc_threshold_bytes={LARGE_ALLOC_THRESHOLD_BYTES}",
        thresholds.par_gemv_min_rows, thresholds.par_tiled_min_rows
    );
    println!("-- asserted: CPU tier {cpu_tier:?} --");
    for (name, m) in &cpu_results {
        println!(
            "   {name:44} thread_allocs={:<4} thread_reallocs={:<4} large_allocs={:<4} \
             large_alloc_max_bytes={}",
            m.thread_allocs, m.thread_reallocs, m.large_allocs, m.large_alloc_max_bytes
        );
    }
    println!("-- reported only: auto-detected tier {auto_tier:?} --");
    for (name, m) in &auto_results {
        println!(
            "   {name:44} thread_allocs={:<4} thread_reallocs={:<4} large_allocs={:<4} \
             large_alloc_max_bytes={}",
            m.thread_allocs, m.thread_reallocs, m.large_allocs, m.large_alloc_max_bytes
        );
    }

    let offenders: Vec<String> = cpu_results
        .iter()
        .filter(|(_, m)| !m.is_clean())
        .map(|(name, m)| {
            format!(
                "{name}: {} thread-local alloc(s), {} thread-local realloc(s), {} large \
                 allocation(s) process-wide (threshold {LARGE_ALLOC_THRESHOLD_BYTES} bytes, max \
                 seen {} bytes)",
                m.thread_allocs, m.thread_reallocs, m.large_allocs, m.large_alloc_max_bytes
            )
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "GEMV paths must perform zero heap allocations on the second call \
         (tier {cpu_tier:?}); offenders:\n{}",
        offenders.join("\n")
    );
}
