//! Selection of, and entry points into, the INT8 dot-product tier (K-14,
//! step 3).
//!
//! # This tier is never a default, and never a silent upgrade
//!
//! Quantizing the activation to int8 changes results bit-for-bit, and this
//! project ships a CPU-vs-Metal **byte-identical** parity guard
//! (`oxibonsai-runtime/tests/cross_backend_determinism_tests.rs`, the
//! `engine_pool` CPU/Metal check, and the README promise those pin). The
//! INT8 tier therefore lives in its own [`Int8Tier`] enum with its own
//! entry points, and is reachable **only** by naming it:
//!
//! - programmatically, by passing an [`Int8Tier`] to one of the `*_int8`
//!   functions below, or
//! - by setting `OXIBONSAI_KERNEL_TIER` to one of [`Int8Tier`]'s names,
//!   which [`Int8Tier::from_env`] reads.
//!
//! [`crate::KernelTier`] is deliberately **not** extended: no `KernelTier`
//! value routes to anything in this module, so `KernelTier::Neon` (and
//! every other existing tier) produces exactly the bits it produced before
//! K-14, and the determinism tests keep passing without being touched or
//! weakened. `int8_tier_parity::the_int8_tier_is_never_selected_implicitly`
//! and [`Int8Tier::from_env`]'s own tests fail loudly if anyone makes this
//! tier reachable without asking for it.
//!
//! # Where the environment selector is read
//!
//! [`Int8Tier::from_env`] is consulted once per call, at the **entry** of
//! every CPU GEMV/GEMM for the formats this tier serves:
//!
//! - the native formats: `KernelDispatcher`'s `OneBitKernel::{gemv, gemm}`
//!   (`Q1_0_g128`) and `TernaryKernel::{gemv_ternary_g128,
//!   gemm_ternary_g128}` (`TQ2_0_g128`), the `parallel_tiled` adaptive
//!   drivers (`gemv_adaptive`, `gemv_adaptive_ternary`,
//!   `gemm_adaptive_ternary`, `gemv_parallel_tiled*`, `gemm_parallel_tiled`)
//!   and the `parallel` drivers (`gemv_1bit_g128_par`, `gemm_1bit_g128_par`,
//!   `gemv_ternary_g128_par`, `gemm_ternary_g128_par`) — see
//!   `KernelDispatcher::native_int8_tier`;
//! - the PrismML formats `PQ2_0` and group-64 `Q2_0`, in
//!   `dispatch_prism.rs`'s `PrismKernel` impl.
//!
//! A driver's tiles and chunks never re-read it: the check happens once,
//! before any strategy is chosen, so the activation is quantized once per
//! call and the INT8 kernels (which parallelize themselves) are never
//! re-entered per tile. (The older `tiled::gemv_tiled` /
//! `tiled::gemv_tiled_par` drivers still call `OneBitKernel::gemv` once
//! per tile, so a direct caller gets the INT8 tier tile by tile — the same
//! values, with the activation re-quantized per tile; nothing in the
//! workspace calls them outside their own tests.) On the native formats a
//! `KernelTier::Gpu`
//! dispatcher and every `*_cached` GPU-handle entry point are never
//! diverted; the PrismML formats have no GPU kernel, so their CPU kernels
//! honour the selector whichever tier the dispatcher is on.
//!
//! # Tiers
//!
//! | [`Int8Tier`] | `OXIBONSAI_KERNEL_TIER` | Requires | Inner loop |
//! |---|---|---|---|
//! | [`Int8Tier::Scalar`] | `int8-scalar` | — | `i32` MAC |
//! | `Int8Tier::Neon` | `neon-int8` | AArch64 | `SMULL` + `SADALP` |
//! | `Int8Tier::NeonDot` | `neon-dot` | `dotprod` | `SDOT` (16 MACs) |
//! | `Int8Tier::NeonI8mm` | `neon-i8mm` | `i8mm` + `dotprod` | `SMMLA` (32 MACs) for GEMM, `SDOT` for GEMV |
//! | `Int8Tier::Avx512Vnni` | `avx512-vnni` | `avx512f/bw/vnni` | `VPDPBUSD` (64 MACs) |
//!
//! Every tier computes the **same `i32`** per block — integer arithmetic is
//! exact and associative, so the only thing that changes between them is
//! how fast the sum is formed. The `f32` result is therefore identical
//! across tiers as well, which `int8_tier_parity` asserts bit-for-bit.
//!
//! # Parallelism
//!
//! Every GEMV splits the weight-row dimension across Rayon tasks above
//! [`INT8_PAR_MIN_ROWS`]; every row-kernel GEMM chooses between a
//! per-batch-row loop of those GEMVs (small `m`, so a 2-token batch still
//! uses every core) and slabs of whole batch rows per task (see
//! `Int8GemmSplit`); the `SMMLA` GEMM splits weight-row pairs. A split only
//! decides *which thread* evaluates an output element, never in what order
//! its terms are summed, so every split is bit-identical to a sequential
//! sweep — the sweep tests below assert it with `to_bits()`.
//!
//! [`Int8Tier::clamp_to_cpu`] mirrors
//! `KernelDispatcher::clamp_tier_to_cpu` (KERN-SOUND / K-03): a tier this
//! CPU cannot execute is demoted, with a warning, rather than being invoked
//! and trapping — so every safe call in this module is safe on every host.

use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::error::{KernelError, KernelResult};
use crate::quant_activation::{Int8Activation, Int8Layout};
use crate::simd_dot_int8::{
    biased_lut16, block_dot_one_bit_scalar, block_dot_two_bit_scalar, Int8TwoBitBlock,
};

/// An INT8 dot-product implementation tier.
///
/// Ordered slowest-to-fastest, like [`crate::KernelTier`], but a separate
/// type — see this module's doc for why it must not be a `KernelTier`
/// variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Int8Tier {
    /// Portable `i32` accumulation — the parity oracle, and the tier every
    /// other one is validated against.
    Scalar,
    /// NEON widening multiply-accumulate (`SMULL` + `SADALP`); ARMv8.0, so
    /// available on every AArch64 host.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// ARMv8.2 `SDOT` — 16 int8 MACs per instruction.
    #[cfg(target_arch = "aarch64")]
    NeonDot,
    /// ARMv8.6 `SMMLA` — 32 int8 MACs per instruction.
    ///
    /// Every GEMM at `m >= 2`, 2-bit and 1-bit alike, runs a decode-reuse
    /// `SMMLA` kernel (`gemm_two_bit_i8mm` / `gemm_one_bit_i8mm`): each
    /// weight-row pair's block is decoded once and applied to every
    /// batch-row pair, where the `SDOT` row kernel re-decodes every weight
    /// row once per batch row. A GEMV (`m == 1`) is a single activation row,
    /// which cannot fill `SMMLA`'s 2x2 tile, so it runs the `SDOT` row
    /// kernel exactly as [`Self::NeonDot`] does. Both kernels form the same
    /// integers, so the output is bit-identical to every other tier's.
    #[cfg(target_arch = "aarch64")]
    NeonI8mm,
    /// AVX-512 VNNI `VPDPBUSD` — 64 int8 MACs per instruction.
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
}

impl std::fmt::Display for Int8Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The environment variable that opts a process into a kernel tier by name.
///
/// Only the INT8 tiers are selectable this way today; an unknown or empty
/// value selects nothing (and is **not** an error, so an unrelated value
/// cannot make a process fail to start).
pub const KERNEL_TIER_ENV: &str = "OXIBONSAI_KERNEL_TIER";

/// Deduplicates [`Int8Tier::clamp_to_cpu`]'s demotion warning to once per
/// process — see that function's doc comment. A production concern (the
/// selector is read on every GEMV/GEMM call), not a test-only one.
static TIER_DEMOTION_WARNED: std::sync::Once = std::sync::Once::new();

/// Serializes every test in this crate's `--lib` unit-test binary that
/// mutates [`KERNEL_TIER_ENV`].
///
/// `std::env::set_var`/`remove_var` are `unsafe fn` (edition 2024) precisely
/// because a concurrent `std::env::var` on *any* key can observe a torn
/// `environ` while another thread mutates it — a whole-process hazard, not a
/// same-key one. Every test here and in `dispatch_prism.rs` that sets the
/// variable does so through a [`TierEnvGuard`], which holds this lock.
///
/// Holding the lock only serializes the *writers*. What keeps the readers
/// deterministic is the per-thread gate in [`Int8Tier::from_env`]: in this
/// crate's own unit-test build the variable is honoured only on a thread
/// that holds a [`TierEnvGuard`], so the many unguarded tests that call the
/// native entry points (`parallel_tests.rs`, `tiled.rs`, `gemm_onebit.rs`,
/// `gemm_ternary.rs`, all comparing raw `f32` bits) can never observe
/// another test's value mid-run.
///
/// `#[cfg(test)]`, and referenced from `dispatch_prism.rs`'s own
/// `#[cfg(test)]` modules — a legal cross-module reference: `cargo test -p
/// oxibonsai-kernels` rebuilds this whole crate with `cfg(test)` active for
/// the one `--lib` test binary. It adds zero surface to the non-test build.
#[cfg(test)]
pub(crate) static KERNEL_TIER_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
thread_local! {
    /// Whether the current thread holds a [`TierEnvGuard`] — the per-thread
    /// gate [`Int8Tier::from_env`] consults in this crate's unit-test build
    /// (see [`KERNEL_TIER_ENV_LOCK`]).
    static TIER_ENV_GUARD_HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// RAII guard around [`KERNEL_TIER_ENV_LOCK`]: [`Self::acquire`] takes the
/// lock, snapshots whatever [`KERNEL_TIER_ENV`] currently holds — a
/// developer's own shell export, or nothing — clears it, and opens the
/// per-thread gate [`Int8Tier::from_env`] checks, so a guarded test always
/// starts from a known-clean environment and is the only thread whose
/// dispatcher calls can see a value it sets. [`Drop::drop`] closes the gate
/// and restores that exact snapshot, including when the guarded test body
/// panics (`Drop` still runs while unwinding), so a failed `assert!`
/// between a test's own `set_var` and its cleanup never leaks the mutated
/// value to a later holder of the lock.
///
/// Tests that toggle [`KERNEL_TIER_ENV`] *mid-body* (to compare the tier
/// selected against the tier cleared, in one test) still do that toggling
/// by hand — that is load-bearing, not cleanup, and this guard does not try
/// to replace it. It only owns the environment's state at entry and at
/// exit.
#[cfg(test)]
pub(crate) struct TierEnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prior: Option<String>,
}

#[cfg(test)]
impl TierEnvGuard {
    pub(crate) fn acquire() -> Self {
        let lock = KERNEL_TIER_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var(KERNEL_TIER_ENV).ok();
        // SAFETY: `lock` (held here, and for the lifetime of the `Self` it
        // moves into) serializes every writer of `KERNEL_TIER_ENV` in this
        // crate's test binary — see `KERNEL_TIER_ENV_LOCK`'s doc comment.
        unsafe {
            std::env::remove_var(KERNEL_TIER_ENV);
        }
        TIER_ENV_GUARD_HELD.with(|held| held.set(true));
        Self { _lock: lock, prior }
    }
}

#[cfg(test)]
impl Drop for TierEnvGuard {
    fn drop(&mut self) {
        TIER_ENV_GUARD_HELD.with(|held| held.set(false));
        // SAFETY: `self._lock` is held for the entire body of `drop`.
        unsafe {
            match &self.prior {
                Some(v) => std::env::set_var(KERNEL_TIER_ENV, v),
                None => std::env::remove_var(KERNEL_TIER_ENV),
            }
        }
    }
}

impl Int8Tier {
    /// The tier's `OXIBONSAI_KERNEL_TIER` spelling.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "int8-scalar",
            #[cfg(target_arch = "aarch64")]
            Self::Neon => "neon-int8",
            #[cfg(target_arch = "aarch64")]
            Self::NeonDot => "neon-dot",
            #[cfg(target_arch = "aarch64")]
            Self::NeonI8mm => "neon-i8mm",
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Vnni => "avx512-vnni",
        }
    }

    /// Parse a tier from its [`Self::name`] (surrounding whitespace, ASCII
    /// case and `-`/`_` are ignored). `None` for anything else.
    ///
    /// Allocation-free: [`Self::from_env`] runs it on every native GEMV/GEMM
    /// call while a tier is selected.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let wanted = name.trim().as_bytes();
        Self::ALL.iter().copied().find(|tier| {
            let canonical = tier.name().as_bytes();
            canonical.len() == wanted.len()
                && canonical.iter().zip(wanted).all(|(&c, &w)| {
                    let w = w.to_ascii_lowercase();
                    c == if w == b'_' { b'-' } else { w }
                })
        })
    }

    /// Every tier this build knows about, slowest first.
    pub const ALL: &'static [Self] = &[
        Self::Scalar,
        #[cfg(target_arch = "aarch64")]
        Self::Neon,
        #[cfg(target_arch = "aarch64")]
        Self::NeonDot,
        #[cfg(target_arch = "aarch64")]
        Self::NeonI8mm,
        #[cfg(target_arch = "x86_64")]
        Self::Avx512Vnni,
    ];

    /// Whether this host can actually execute the tier.
    #[must_use]
    pub fn is_supported(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            Self::Neon => true,
            #[cfg(target_arch = "aarch64")]
            Self::NeonDot => std::arch::is_aarch64_feature_detected!("dotprod"),
            // `i8mm` **and** `dotprod`: this tier's GEMV runs on `SDOT`.
            // Both are mandatory from Armv8.6 (where `FEAT_I8MM` becomes
            // mandatory, and `FEAT_DotProd` has been since Armv8.4), but an
            // Armv8.2 implementation may in principle carry `i8mm` alone —
            // requiring both here keeps every code path this tier can reach
            // executable.
            #[cfg(target_arch = "aarch64")]
            Self::NeonI8mm => {
                std::arch::is_aarch64_feature_detected!("i8mm")
                    && std::arch::is_aarch64_feature_detected!("dotprod")
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Vnni => {
                is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512vnni")
            }
        }
    }

    /// Demote a tier this CPU cannot execute to the best one it can — the
    /// INT8 twin of `KernelDispatcher::clamp_tier_to_cpu` (K-03/sec-14), and
    /// what makes every safe entry point in this module safe on every host.
    ///
    /// The demotion itself happens on every call, but its `tracing::warn!`
    /// fires once per process: the environment selector is read on every
    /// GEMV/GEMM call by design (a cached value would go stale the moment
    /// the variable changes), so an unsupported-tier request would otherwise
    /// log hundreds of times per decoded token.
    #[must_use]
    pub fn clamp_to_cpu(self) -> Self {
        if self.is_supported() {
            return self;
        }
        let demoted = Self::best_available();
        TIER_DEMOTION_WARNED.call_once(|| {
            tracing::warn!(
                requested = %self,
                demoted_to = %demoted,
                "requested INT8 kernel tier not supported by this CPU, demoting \
                 (further demotions this process will not be logged)"
            );
        });
        demoted
    }

    /// The fastest tier this host supports. Never consulted implicitly —
    /// callers still have to ask for the INT8 tier in the first place.
    #[must_use]
    pub fn best_available() -> Self {
        let mut best = Self::Scalar;
        for tier in Self::ALL {
            if tier.is_supported() {
                best = *tier;
            }
        }
        best
    }

    /// The tier `OXIBONSAI_KERNEL_TIER` asks for, if any.
    ///
    /// **`None` when the variable is unset** — that is the guard that keeps
    /// this tier off by default. A named-but-unsupported tier is clamped
    /// (with a warning) rather than rejected, so a config that travels
    /// between machines still runs. An unknown name selects nothing.
    ///
    /// # In this crate's own unit-test build
    ///
    /// The variable is honoured only on a thread that holds a
    /// `TierEnvGuard` (every other thread sees `None`), so one test that
    /// sets it can never change the bits another, unguarded test computes on
    /// a concurrent thread. Integration tests and every downstream crate
    /// link the regular build, where every thread reads the variable.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        #[cfg(test)]
        if !TIER_ENV_GUARD_HELD.with(std::cell::Cell::get) {
            return None;
        }
        let raw = std::env::var(KERNEL_TIER_ENV).ok()?;
        Some(Self::from_name(&raw)?.clamp_to_cpu())
    }
}

/// Whether this tier's NEON dot should use `SDOT`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn uses_sdot(tier: Int8Tier) -> bool {
    matches!(tier, Int8Tier::NeonDot | Int8Tier::NeonI8mm)
}

/// One block's biased int8 dot, on `tier`.
///
/// `qs` is the weight block's packed codes and `act` the matching
/// stride-4-permuted activation slice. The bias is removed by the caller.
#[inline]
fn two_bit_block_dot(tier: Int8Tier, qs: &[u8], act: &[i8], lut16: &[u8; 16]) -> i32 {
    match tier {
        Int8Tier::Scalar => {
            let lut = [lut16[0], lut16[1], lut16[2], lut16[3]];
            block_dot_two_bit_scalar(qs, act, &lut)
        }
        #[cfg(target_arch = "aarch64")]
        _ => {
            // SAFETY: NEON is the AArch64 baseline, and the `SDOT` variant
            // is only chosen for a tier `is_supported()` confirmed has
            // `dotprod`.
            unsafe {
                let lv = crate::simd_dot_int8::load_lut(lut16);
                if uses_sdot(tier) {
                    crate::simd_dot_int8::block_dot_two_bit_neon::<true>(qs, act, lv)
                } else {
                    crate::simd_dot_int8::block_dot_two_bit_neon::<false>(qs, act, lv)
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        Int8Tier::Avx512Vnni => {
            // SAFETY: `is_supported()` confirmed avx512f + avx512bw +
            // avx512vnni before this tier could be selected.
            unsafe { crate::simd_dot_int8::block_dot_two_bit_avx512vnni(qs, act, lut16) }
        }
    }
}

/// One 1-bit block's biased int8 dot, on `tier`.
#[inline]
fn one_bit_block_dot(tier: Int8Tier, qs: &[u8], act: &[i8]) -> i32 {
    match tier {
        Int8Tier::Scalar => block_dot_one_bit_scalar(qs, act),
        #[cfg(target_arch = "aarch64")]
        _ => {
            // SAFETY: see `two_bit_block_dot`.
            unsafe {
                if uses_sdot(tier) {
                    crate::simd_dot_int8::block_dot_one_bit_neon::<true>(qs, act)
                } else {
                    crate::simd_dot_int8::block_dot_one_bit_neon::<false>(qs, act)
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        Int8Tier::Avx512Vnni => {
            // SAFETY: see `two_bit_block_dot`.
            unsafe { crate::simd_dot_int8::block_dot_one_bit_avx512vnni(qs, act) }
        }
    }
}

/// Validate a GEMV/GEMM's shapes and return `blocks_per_row`.
///
/// Checks, in order and with the same buffer names, exactly what the f32
/// drivers in `parallel.rs` / `parallel_tiled.rs` check, so diverting a call
/// to this tier never changes which error a caller sees.
fn validate_int8(
    n_blocks: usize,
    input_len: usize,
    output_len: usize,
    m: usize,
    n_rows: usize,
    k: usize,
    qk: usize,
) -> KernelResult<usize> {
    if !k.is_multiple_of(qk) {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: qk,
        });
    }
    if input_len < m * k {
        return Err(KernelError::dimension_mismatch("input", m * k, input_len));
    }
    if output_len < m * n_rows {
        return Err(KernelError::buffer_too_small(
            "output",
            m * n_rows,
            output_len,
        ));
    }
    let blocks_per_row = k / qk;
    let expected = n_rows * blocks_per_row;
    if n_blocks < expected {
        return Err(KernelError::buffer_too_small("blocks", expected, n_blocks));
    }
    Ok(blocks_per_row)
}

/// One activation row's quantized form, as the row kernels consume it.
///
/// Bundled into a struct rather than passed as three more parameters so
/// the row kernels stay inside clippy's `too_many_arguments` budget.
#[derive(Debug, Clone, Copy)]
struct Int8Row<'a> {
    /// The row's int8 codes, in the kernel's expected layout.
    codes: &'a [i8],
    /// Per-block activation scales.
    scales: &'a [f32],
    /// Per-block exact code sums (the bias correction).
    sums: &'a [i32],
}

impl<'a> Int8Row<'a> {
    /// Batch row `r` of a quantized activation.
    fn of(act: &'a Int8Activation, r: usize) -> Self {
        Self {
            codes: act.codes_row(r),
            scales: act.scales_row(r),
            sums: act.sums_row(r),
        }
    }
}

/// `output[row] = Σ_b d_b * s_b * (biased_dot_b − sum_b)` for one
/// activation row over `output.len()` weight rows — the whole 2-bit INT8
/// GEMV, minus the parallel split.
fn two_bit_rows<B: Int8TwoBitBlock>(
    tier: Int8Tier,
    blocks: &[B],
    row: Int8Row<'_>,
    output: &mut [f32],
    blocks_per_row: usize,
    lut16: &[u8; 16],
) {
    let Int8Row {
        codes,
        scales,
        sums,
    } = row;
    for (ni, out) in output.iter_mut().enumerate() {
        let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
        let mut sum = 0.0f32;
        for (bi, block) in row_blocks.iter().enumerate() {
            let act = &codes[bi * B::QK..(bi + 1) * B::QK];
            let acc = two_bit_block_dot(tier, block.packed_codes(), act, lut16) - sums[bi];
            sum += block.block_scale() * scales[bi] * acc as f32;
        }
        *out = sum;
    }
}

/// The 1-bit twin of [`two_bit_rows`]: same per-block arithmetic, same
/// summation order, on `Q1_0_g128` sign bits.
fn one_bit_rows(
    tier: Int8Tier,
    blocks: &[BlockQ1_0G128],
    row: Int8Row<'_>,
    output: &mut [f32],
    blocks_per_row: usize,
) {
    let Int8Row {
        codes,
        scales,
        sums,
    } = row;
    for (ni, out) in output.iter_mut().enumerate() {
        let row_blocks = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
        let mut sum = 0.0f32;
        for (bi, block) in row_blocks.iter().enumerate() {
            let act = &codes[bi * QK1_0_G128..(bi + 1) * QK1_0_G128];
            let acc = one_bit_block_dot(tier, &block.qs, act) - sums[bi];
            sum += block.d.to_f32() * scales[bi] * acc as f32;
        }
        *out = sum;
    }
}

/// Weight rows per Rayon task for the INT8 GEMV — the same shape
/// `parallel.rs::rows_per_task` uses (K-16).
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn int8_rows_per_task(n_rows: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    (n_rows / (threads * 4)).clamp(8, 512)
}

/// Weight-row count below which an INT8 GEMV stays on one thread.
pub const INT8_PAR_MIN_ROWS: usize = 256;

/// Evaluate `rows(block_slice, out_chunk)` over every weight row of
/// `output` (one entry per weight row), split into Rayon chunks of whole
/// weight rows once there are at least [`INT8_PAR_MIN_ROWS`] of them.
///
/// `rows` must compute each output element from its own weight row alone
/// (both row kernels do), which is what makes the split bit-exact.
fn for_weight_rows<B, F>(blocks: &[B], output: &mut [f32], blocks_per_row: usize, rows: F)
where
    B: Sync,
    F: Fn(&[B], &mut [f32]) + Sync,
{
    let n_rows = output.len();
    #[cfg(not(target_arch = "wasm32"))]
    if n_rows >= INT8_PAR_MIN_ROWS {
        let chunk = int8_rows_per_task(n_rows);
        output
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out_chunk)| {
                let row_start = ci * chunk;
                let slice = &blocks
                    [row_start * blocks_per_row..(row_start + out_chunk.len()) * blocks_per_row];
                rows(slice, out_chunk);
            });
        return;
    }
    rows(&blocks[..n_rows * blocks_per_row], output);
}

/// How an INT8 GEMM spreads its `m` batch rows over Rayon (WASM has no
/// worker pool, so its GEMM is always a sequential batch-row loop).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Int8GemmSplit {
    /// One weight-row-parallel GEMV per batch row. Used when there are
    /// fewer batch rows than threads (a batch split would leave cores idle —
    /// `m = 2` on an 8-core host would use 2 of them) and enough weight rows
    /// for [`for_weight_rows`] to split.
    PerRowGemv,
    /// Slabs of whole batch rows per Rayon task, each evaluated over every
    /// weight row.
    BatchSlabs,
}

/// The split [`int8_gemm_rows`] uses for an `m x n_rows` GEMM on a pool of
/// `threads` workers. Pure, so the boundaries are testable on any host.
#[cfg(not(target_arch = "wasm32"))]
fn int8_gemm_split(m: usize, n_rows: usize, threads: usize) -> Int8GemmSplit {
    if m < threads && n_rows >= INT8_PAR_MIN_ROWS {
        Int8GemmSplit::PerRowGemv
    } else {
        Int8GemmSplit::BatchSlabs
    }
}

/// Batch rows per Rayon task for [`Int8GemmSplit::BatchSlabs`].
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn int8_gemm_rows_per_task(m: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    m.div_ceil(threads).max(1).min(m)
}

/// Batch-row count below which [`Int8GemmSplit::BatchSlabs`] stays on one
/// thread.
#[cfg(not(target_arch = "wasm32"))]
const INT8_GEMM_PAR_MIN_BATCH: usize = 2;

/// The INT8 GEMM driver shared by both formats.
///
/// `batch_row(mi, out_row, parallel)` must write batch row `mi`'s `n_rows`
/// outputs into `out_row`, splitting its weight rows across Rayon when
/// `parallel` is true (via [`for_weight_rows`]) and staying on the calling
/// thread otherwise. Every output element is computed by the same row
/// kernel with the same per-block order whichever split runs it.
fn int8_gemm_rows<F>(output: &mut [f32], m: usize, n_rows: usize, batch_row: F)
where
    F: Fn(usize, &mut [f32], bool) + Sync,
{
    let output = &mut output[..m * n_rows];

    #[cfg(target_arch = "wasm32")]
    {
        for (mi, out_row) in output.chunks_mut(n_rows).enumerate() {
            batch_row(mi, out_row, false);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = rayon::current_num_threads().max(1);
        match int8_gemm_split(m, n_rows, threads) {
            Int8GemmSplit::PerRowGemv => {
                for (mi, out_row) in output.chunks_mut(n_rows).enumerate() {
                    batch_row(mi, out_row, true);
                }
            }
            Int8GemmSplit::BatchSlabs => {
                if m < INT8_GEMM_PAR_MIN_BATCH {
                    for (mi, out_row) in output.chunks_mut(n_rows).enumerate() {
                        batch_row(mi, out_row, false);
                    }
                    return;
                }
                let chunk = int8_gemm_rows_per_task(m);
                output
                    .par_chunks_mut(chunk * n_rows)
                    .enumerate()
                    .for_each(|(ci, out_chunk)| {
                        let m0 = ci * chunk;
                        for (r, out_row) in out_chunk.chunks_mut(n_rows).enumerate() {
                            batch_row(m0 + r, out_row, false);
                        }
                    });
            }
        }
    }
}

/// INT8 GEMV for any 2-bit format: `output[row] = dot(weight_row, input)`.
///
/// The activation is quantized **once** here and reused by all `n_rows`
/// weight rows (K-14, step 1), which is what makes the tier pay; the weight
/// rows are then split across Rayon above [`INT8_PAR_MIN_ROWS`].
///
/// # Errors
///
/// - [`KernelError::NotBlockAligned`] if `k` is not a multiple of the
///   format's block size.
/// - [`KernelError::NamedDimensionMismatch`] if `input` is shorter than `k`.
/// - [`KernelError::NamedBufferTooSmall`] if `output` or `blocks` is too
///   short.
pub fn gemv_two_bit_int8<B: Int8TwoBitBlock + Sync>(
    tier: Int8Tier,
    blocks: &[B],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let tier = tier.clamp_to_cpu();
    let blocks_per_row =
        validate_int8(blocks.len(), input.len(), output.len(), 1, n_rows, k, B::QK)?;
    if n_rows == 0 {
        return Ok(());
    }
    let act = Int8Activation::quantize(input, 1, k, B::QK, Int8Layout::Stride4)?;
    let lut16 = biased_lut16(&B::BIASED_LUT);
    let row = Int8Row::of(&act, 0);
    for_weight_rows(blocks, &mut output[..n_rows], blocks_per_row, |blk, out| {
        two_bit_rows(tier, blk, row, out, blocks_per_row, &lut16);
    });
    Ok(())
}

/// INT8 GEMM for any 2-bit format: `output[m, n] = dot(weight_n, input_m)`.
///
/// The whole `m x k` activation is quantized once, then spread over Rayon
/// by `int8_gemm_rows`: a per-batch-row loop of weight-row-parallel GEMVs
/// while `m` is below the thread count (so a 2-token batch still uses every
/// core), slabs of batch rows otherwise. `m == 1` is exactly
/// [`gemv_two_bit_int8`], bit for bit.
///
/// `Int8Tier::NeonI8mm` at `m >= 2` runs `gemm_two_bit_i8mm` instead — see
/// the variant's doc comment. Every path yields the same bits.
///
/// # Errors
///
/// See [`gemv_two_bit_int8`].
pub fn gemm_two_bit_int8<B: Int8TwoBitBlock + Sync>(
    tier: Int8Tier,
    blocks: &[B],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let tier = tier.clamp_to_cpu();
    let blocks_per_row =
        validate_int8(blocks.len(), input.len(), output.len(), m, n_rows, k, B::QK)?;
    if m == 0 || n_rows == 0 {
        return Ok(());
    }
    let act = Int8Activation::quantize(input, m, k, B::QK, Int8Layout::Stride4)?;
    let lut16 = biased_lut16(&B::BIASED_LUT);

    #[cfg(target_arch = "aarch64")]
    if tier == Int8Tier::NeonI8mm && m >= 2 {
        gemm_two_bit_i8mm(blocks, &act, output, m, n_rows, blocks_per_row, &lut16);
        return Ok(());
    }

    int8_gemm_rows(output, m, n_rows, |mi, out_row, parallel| {
        let row = Int8Row::of(&act, mi);
        if parallel {
            for_weight_rows(blocks, out_row, blocks_per_row, |blk, out| {
                two_bit_rows(tier, blk, row, out, blocks_per_row, &lut16);
            });
        } else {
            two_bit_rows(tier, blocks, row, out_row, blocks_per_row, &lut16);
        }
    });
    Ok(())
}

/// The activation side of the `SMMLA` GEMM: every batch-row pair's codes,
/// interleaved once per call into the `B`-operand layout the weight-pair
/// decoders produce (`decode_two_bit_pair_for_mmla` for the stride-4 2-bit
/// layout, `decode_one_bit_pair_for_mmla` for the sequential 1-bit one),
/// plus each pair's per-block scales and code sums spread over the four
/// tile lanes.
///
/// Block-major (`[block][pair]`), so the inner loop over batch-row pairs of
/// one block reads one contiguous stream. An odd `m` pairs its last row
/// with itself; that duplicate's lanes are computed and discarded, so no
/// scalar tail path exists to drift.
#[cfg(target_arch = "aarch64")]
struct I8mmPanel {
    /// `blocks_per_row * pairs * qk * 2` interleaved codes.
    codes: Vec<i8>,
    /// `[s0, s1, s0, s1]` per `(block, pair)` — the tile's lane order.
    scales: Vec<f32>,
    /// `[sum0, sum1, sum0, sum1]` per `(block, pair)`.
    sums: Vec<i32>,
    pairs: usize,
    /// Interleaved code bytes per `(block, pair)`: `qk * 2`.
    block_bytes: usize,
}

#[cfg(target_arch = "aarch64")]
impl I8mmPanel {
    /// Interleave `act`'s `m` rows (blocks of `qk`, in `act`'s own layout).
    fn build(act: &Int8Activation, m: usize, qk: usize, blocks_per_row: usize) -> Self {
        let pairs = m.div_ceil(2);
        let chunks = qk / 8;
        let block_bytes = chunks * 16;
        let layout = act.layout();
        let mut codes = vec![0i8; blocks_per_row * pairs * block_bytes];
        let mut scales = vec![0.0f32; blocks_per_row * pairs * 4];
        let mut sums = vec![0i32; blocks_per_row * pairs * 4];
        for p in 0..pairs {
            let (r0, r1) = (2 * p, (2 * p + 1).min(m - 1));
            let (x0, x1) = (act.codes_row(r0), act.codes_row(r1));
            let (s0, s1) = (act.scales_row(r0), act.scales_row(r1));
            let (u0, u1) = (act.sums_row(r0), act.sums_row(r1));
            for b in 0..blocks_per_row {
                let base = (b * pairs + p) * block_bytes;
                for c in 0..chunks {
                    // Chunk `c`'s eight activations, in the order the weight
                    // decoder emits chunk `c`'s eight weights.
                    let within = match layout {
                        Int8Layout::Stride4 => {
                            let (g, s, h) = (c / 8, (c / 2) % 4, c % 2);
                            g * 64 + s * 16 + h * 8
                        }
                        Int8Layout::Sequential => c * 8,
                    };
                    let src = b * qk + within;
                    let dst = base + c * 16;
                    codes[dst..dst + 8].copy_from_slice(&x0[src..src + 8]);
                    codes[dst + 8..dst + 16].copy_from_slice(&x1[src..src + 8]);
                }
                let lane = (b * pairs + p) * 4;
                scales[lane..lane + 4].copy_from_slice(&[s0[b], s1[b], s0[b], s1[b]]);
                sums[lane..lane + 4].copy_from_slice(&[u0[b], u1[b], u0[b], u1[b]]);
            }
        }
        Self {
            codes,
            scales,
            sums,
            pairs,
            block_bytes,
        }
    }

    /// Block `b`'s pair-0 codes, scales and sums — consecutive pairs follow
    /// contiguously (`block_bytes` code bytes, 4 lanes apart).
    fn block(&self, b: usize) -> (&[i8], &[f32], &[i32]) {
        (
            &self.codes[b * self.pairs * self.block_bytes..],
            &self.scales[b * self.pairs * 4..],
            &self.sums[b * self.pairs * 4..],
        )
    }
}

/// Weight rows per Rayon task for the `SMMLA` GEMM: the GEMV's row shape,
/// rounded up to a whole number of weight-row pairs.
#[cfg(target_arch = "aarch64")]
fn i8mm_rows_per_task(n_rows: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let rows = (n_rows / (threads * 4)).clamp(16, 512);
    rows + rows % 2
}

/// The decode-reuse `SMMLA` GEMM driver `Int8Tier::NeonI8mm` runs for
/// every GEMM at `m >= 2`, both formats.
///
/// `pair_block(n0, n1, b, panel, accs)` must decode block `b` of weight
/// rows `n0` and `n1` **once** and fold it against every batch-row pair of
/// `panel` into `accs` (`[n0m0, n0m1, n1m0, n1m1]` per pair) — 16 `SMMLA`s
/// (32 int8 MACs each) per 128 weights per pair, where the `SDOT` row
/// kernel re-decodes every weight row once per batch row. Weight rows are
/// split across Rayon in whole pairs; an odd `n_rows` pairs its last row
/// with itself (duplicate lanes discarded).
///
/// Bit-identical to the row kernels on every cell: each `SMMLA` tile lane
/// is the same exact `i32` the other tiers form, and each cell accumulates
/// `(d * s) * (acc - sum)` over the blocks in the same order with separate
/// multiply and add (never a fused multiply-add) —
/// `i8mm_gemm_matches_the_gemv_sweep_at_every_shape` pins it.
#[cfg(target_arch = "aarch64")]
fn i8mm_gemm<F>(
    panel: &I8mmPanel,
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    blocks_per_row: usize,
    pair_block: F,
) where
    F: Fn(usize, usize, usize, &I8mmPanel, &mut [f32]) + Sync,
{
    let chunk_rows = i8mm_rows_per_task(n_rows);
    let n_tasks = n_rows.div_ceil(chunk_rows);
    // Hand every task its own column range of every batch row: disjoint
    // `&mut` slices, so the split needs no `unsafe`.
    let mut per_task: Vec<Vec<&mut [f32]>> = (0..n_tasks).map(|_| Vec::with_capacity(m)).collect();
    for row in output[..m * n_rows].chunks_mut(n_rows) {
        for (t, piece) in row.chunks_mut(chunk_rows).enumerate() {
            per_task[t].push(piece);
        }
    }
    per_task
        .into_par_iter()
        .enumerate()
        .for_each(|(t, mut out_rows)| {
            i8mm_task(
                panel,
                &mut out_rows,
                t * chunk_rows,
                blocks_per_row,
                &pair_block,
            );
        });
}

/// `Int8Tier::NeonI8mm`'s 2-bit GEMM (`m >= 2`) on [`i8mm_gemm`], decoding
/// with the format's own biased table (`0b11 -> 0` for `TQ2_0_g128`).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn gemm_two_bit_i8mm<B: Int8TwoBitBlock + Sync>(
    blocks: &[B],
    act: &Int8Activation,
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    blocks_per_row: usize,
    lut16: &[u8; 16],
) {
    use crate::simd_dot_int8::{load_lut, mmla_pair_block};
    use core::arch::aarch64::{vcombine_f32, vdup_n_f32};

    let panel = I8mmPanel::build(act, m, B::QK, blocks_per_row);
    let groups = B::QK / 64;
    i8mm_gemm(
        &panel,
        output,
        m,
        n_rows,
        blocks_per_row,
        |n0, n1, b, panel, accs| {
            let (w0, w1) = (
                &blocks[n0 * blocks_per_row + b],
                &blocks[n1 * blocks_per_row + b],
            );
            let (codes, scales, sums) = panel.block(b);
            // SAFETY: only reached for `Int8Tier::NeonI8mm`, which
            // `is_supported()` gates on `i8mm` + `dotprod`; each block holds
            // `groups * 16` code bytes (the block type), and the panel slices
            // hold `panel.pairs` whole entries for block `b`
            // (`I8mmPanel::build`), as `accs` holds `panel.pairs * 4` floats.
            unsafe {
                let lut = load_lut(lut16);
                let d4 = vcombine_f32(vdup_n_f32(w0.block_scale()), vdup_n_f32(w1.block_scale()));
                let (qs0, qs1) = (w0.packed_codes().as_ptr(), w1.packed_codes().as_ptr());
                let (c, s, u) = (codes.as_ptr(), scales.as_ptr(), sums.as_ptr());
                let slot = accs.as_mut_ptr();
                if groups == 2 {
                    mmla_pair_block::<2>(qs0, qs1, lut, d4, c, s, u, panel.pairs, slot);
                } else {
                    mmla_pair_block::<1>(qs0, qs1, lut, d4, c, s, u, panel.pairs, slot);
                }
            }
        },
    );
}

/// `Int8Tier::NeonI8mm`'s 1-bit `Q1_0_g128` GEMM (`m >= 2`) on
/// [`i8mm_gemm`].
#[cfg(target_arch = "aarch64")]
fn gemm_one_bit_i8mm(
    blocks: &[BlockQ1_0G128],
    act: &Int8Activation,
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    blocks_per_row: usize,
) {
    use crate::simd_dot_int8::mmla_pair_block_one_bit;
    use core::arch::aarch64::{vcombine_f32, vdup_n_f32};

    let panel = I8mmPanel::build(act, m, QK1_0_G128, blocks_per_row);
    i8mm_gemm(
        &panel,
        output,
        m,
        n_rows,
        blocks_per_row,
        |n0, n1, b, panel, accs| {
            let (w0, w1) = (
                &blocks[n0 * blocks_per_row + b],
                &blocks[n1 * blocks_per_row + b],
            );
            let (codes, scales, sums) = panel.block(b);
            // SAFETY: only reached for `Int8Tier::NeonI8mm` (`i8mm` +
            // `dotprod`); each block holds 16 sign bytes, and the panel
            // slices / `accs` are sized as in `gemm_two_bit_i8mm`.
            unsafe {
                let d4 = vcombine_f32(vdup_n_f32(w0.d.to_f32()), vdup_n_f32(w1.d.to_f32()));
                mmla_pair_block_one_bit(
                    w0.qs.as_ptr(),
                    w1.qs.as_ptr(),
                    d4,
                    codes.as_ptr(),
                    scales.as_ptr(),
                    sums.as_ptr(),
                    panel.pairs,
                    accs.as_mut_ptr(),
                );
            }
        },
    );
}

/// One Rayon task of [`i8mm_gemm`]: weight rows
/// `n_start .. n_start + out_rows[0].len()` against every batch row.
#[cfg(target_arch = "aarch64")]
fn i8mm_task<F>(
    panel: &I8mmPanel,
    out_rows: &mut [&mut [f32]],
    n_start: usize,
    blocks_per_row: usize,
    pair_block: &F,
) where
    F: Fn(usize, usize, usize, &I8mmPanel, &mut [f32]),
{
    let m = out_rows.len();
    let rows = out_rows.first().map_or(0, |r| r.len());
    // One `[n0m0, n0m1, n1m0, n1m1]` accumulator per batch-row pair.
    let mut accs = vec![0.0f32; panel.pairs * 4];
    let mut ln = 0usize;
    while ln < rows {
        let n0 = n_start + ln;
        let n1 = if ln + 1 < rows { n0 + 1 } else { n0 };
        accs.fill(0.0);
        for b in 0..blocks_per_row {
            pair_block(n0, n1, b, panel, &mut accs);
        }
        for p in 0..panel.pairs {
            let (m0, m1) = (2 * p, 2 * p + 1);
            let a = &accs[p * 4..p * 4 + 4];
            out_rows[m0][ln] = a[0];
            if m1 < m {
                out_rows[m1][ln] = a[1];
            }
            if ln + 1 < rows {
                out_rows[m0][ln + 1] = a[2];
                if m1 < m {
                    out_rows[m1][ln + 1] = a[3];
                }
            }
        }
        ln += 2;
    }
}

/// INT8 GEMV for the 1-bit `Q1_0_g128` format, weight rows split across
/// Rayon above [`INT8_PAR_MIN_ROWS`] exactly like [`gemv_two_bit_int8`].
///
/// # Errors
///
/// See [`gemv_two_bit_int8`].
pub fn gemv_1bit_g128_int8(
    tier: Int8Tier,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let tier = tier.clamp_to_cpu();
    let blocks_per_row = validate_int8(
        blocks.len(),
        input.len(),
        output.len(),
        1,
        n_rows,
        k,
        QK1_0_G128,
    )?;
    if n_rows == 0 {
        return Ok(());
    }
    let act = Int8Activation::quantize(input, 1, k, QK1_0_G128, Int8Layout::Sequential)?;
    let row = Int8Row::of(&act, 0);
    for_weight_rows(blocks, &mut output[..n_rows], blocks_per_row, |blk, out| {
        one_bit_rows(tier, blk, row, out, blocks_per_row);
    });
    Ok(())
}

/// INT8 GEMM for the 1-bit `Q1_0_g128` format, spread over Rayon by the
/// same `int8_gemm_rows` driver as [`gemm_two_bit_int8`]; `m == 1` is
/// exactly [`gemv_1bit_g128_int8`], bit for bit. `Int8Tier::NeonI8mm` at
/// `m >= 2` runs the decode-reuse `SMMLA` kernel (`gemm_one_bit_i8mm`) —
/// the same bits.
///
/// # Errors
///
/// See [`gemv_two_bit_int8`].
pub fn gemm_1bit_g128_int8(
    tier: Int8Tier,
    blocks: &[BlockQ1_0G128],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
) -> KernelResult<()> {
    let tier = tier.clamp_to_cpu();
    let blocks_per_row = validate_int8(
        blocks.len(),
        input.len(),
        output.len(),
        m,
        n_rows,
        k,
        QK1_0_G128,
    )?;
    if m == 0 || n_rows == 0 {
        return Ok(());
    }
    let act = Int8Activation::quantize(input, m, k, QK1_0_G128, Int8Layout::Sequential)?;

    #[cfg(target_arch = "aarch64")]
    if tier == Int8Tier::NeonI8mm && m >= 2 {
        gemm_one_bit_i8mm(blocks, &act, output, m, n_rows, blocks_per_row);
        return Ok(());
    }

    int8_gemm_rows(output, m, n_rows, |mi, out_row, parallel| {
        let row = Int8Row::of(&act, mi);
        if parallel {
            for_weight_rows(blocks, out_row, blocks_per_row, |blk, out| {
                one_bit_rows(tier, blk, row, out, blocks_per_row);
            });
        } else {
            one_bit_rows(tier, blocks, row, out_row, blocks_per_row);
        }
    });
    Ok(())
}

#[cfg(test)]
mod int8_dispatch_tests {
    use super::*;
    use oxibonsai_core::{BlockPQ2_0, BlockTQ2_0_g128, QK_PQ2_0, QK_TQ2_0_G128};

    /// Deterministic LCG for synthetic weights and activations.
    struct Lcg(u32);

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self(seed | 1)
        }
        fn next_u8(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 19) as u8
        }
        fn next_f32(&mut self) -> f32 {
            (i32::from(self.next_u8()) - 128) as f32 / 64.0
        }
    }

    fn pq2_blocks(n: usize, seed: u32) -> Vec<BlockPQ2_0> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let mut qs = [0u8; 32];
                for b in &mut qs {
                    *b = rng.next_u8();
                }
                BlockPQ2_0 {
                    d: half::f16::from_f32(0.0625 + (rng.next_u8() % 16) as f32 / 256.0),
                    qs,
                }
            })
            .collect()
    }

    fn tq2_blocks(n: usize, seed: u32) -> Vec<BlockTQ2_0_g128> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let mut qs = [0u8; 32];
                for b in &mut qs {
                    *b = rng.next_u8();
                }
                BlockTQ2_0_g128 {
                    qs,
                    d: half::f16::from_f32(0.0625 + (rng.next_u8() % 16) as f32 / 256.0),
                }
            })
            .collect()
    }

    fn q1_blocks(n: usize, seed: u32) -> Vec<BlockQ1_0G128> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let mut qs = [0u8; QK1_0_G128 / 8];
                for b in &mut qs {
                    *b = rng.next_u8();
                }
                BlockQ1_0G128 {
                    d: half::f16::from_f32(0.125 + (rng.next_u8() % 16) as f32 / 256.0),
                    qs,
                }
            })
            .collect()
    }

    fn activations(len: usize, seed: u32) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..len).map(|_| rng.next_f32()).collect()
    }

    fn supported_tiers() -> Vec<Int8Tier> {
        Int8Tier::ALL
            .iter()
            .copied()
            .filter(|t| t.is_supported())
            .collect()
    }

    fn assert_bits_eq(expect: &[f32], got: &[f32], what: &str) {
        assert_eq!(expect.len(), got.len(), "{what}: length");
        for (i, (e, g)) in expect.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                e.to_bits(),
                g.to_bits(),
                "{what}: cell {i} diverged: {e} vs {g}"
            );
        }
    }

    /// The batch sizes every GEMM sweep below covers: `1`, the smallest
    /// split, both sides of the thread-count boundary [`int8_gemm_split`]
    /// uses, `PRISM_GEMM_MR`, and an odd size past every boundary.
    fn boundary_batches() -> Vec<usize> {
        let threads = rayon::current_num_threads().max(1);
        let mut ms = vec![
            1,
            2,
            threads.saturating_sub(1).max(1),
            threads,
            threads + 1,
            crate::dequant_prism::PRISM_GEMM_MR,
            13,
        ];
        ms.sort_unstable();
        ms.dedup();
        ms
    }

    /// The guard the HARD CONSTRAINT asks for: with the environment clean,
    /// nothing selects this tier.
    ///
    /// [`TierEnvGuard`] both takes [`KERNEL_TIER_ENV_LOCK`] and guarantees a
    /// clean environment on entry and a restored one on exit (even on
    /// panic), so this test needs no manual `remove_var` of its own at
    /// either end.
    #[test]
    fn from_env_selects_nothing_unless_asked_and_then_exactly_what_was_asked() {
        let _guard = TierEnvGuard::acquire();
        assert_eq!(
            Int8Tier::from_env(),
            None,
            "the INT8 tier must never be selected with a clean environment"
        );

        unsafe {
            std::env::set_var(KERNEL_TIER_ENV, "int8-scalar");
        }
        assert_eq!(Int8Tier::from_env(), Some(Int8Tier::Scalar));

        unsafe {
            std::env::set_var(KERNEL_TIER_ENV, "  INT8_SCALAR  ");
        }
        assert_eq!(
            Int8Tier::from_env(),
            Some(Int8Tier::Scalar),
            "names are case- and separator-insensitive"
        );

        unsafe {
            std::env::set_var(KERNEL_TIER_ENV, "not-a-tier");
        }
        assert_eq!(
            Int8Tier::from_env(),
            None,
            "an unknown tier name must select nothing, not fail"
        );
        // `_guard`'s `Drop` restores the environment from here, no manual
        // cleanup needed.
    }

    /// The unit-test isolation gate: while one thread holds a
    /// [`TierEnvGuard`] and has the variable set, every other thread — the
    /// unguarded tests elsewhere in this binary that compare the native
    /// entry points' raw bits — must still read `None`.
    #[test]
    fn from_env_is_honoured_only_on_the_thread_holding_the_guard() {
        let _guard = TierEnvGuard::acquire();
        unsafe {
            std::env::set_var(KERNEL_TIER_ENV, "int8-scalar");
        }
        assert_eq!(Int8Tier::from_env(), Some(Int8Tier::Scalar));
        let other_thread = std::thread::scope(|s| {
            s.spawn(Int8Tier::from_env)
                .join()
                .unwrap_or(Some(Int8Tier::Scalar))
        });
        assert_eq!(
            other_thread, None,
            "a thread without a TierEnvGuard must not see another test's tier"
        );
    }

    #[test]
    fn every_tier_round_trips_through_its_name() {
        for tier in Int8Tier::ALL {
            assert_eq!(Int8Tier::from_name(tier.name()), Some(*tier));
            assert_eq!(tier.to_string(), tier.name());
        }
    }

    #[test]
    fn best_available_is_supported_and_scalar_always_is() {
        assert!(Int8Tier::Scalar.is_supported());
        assert!(Int8Tier::best_available().is_supported());
        assert!(Int8Tier::best_available() >= Int8Tier::Scalar);
    }

    #[test]
    fn clamp_to_cpu_never_returns_an_unsupported_tier() {
        for tier in Int8Tier::ALL {
            assert!(tier.clamp_to_cpu().is_supported());
        }
    }

    #[test]
    fn gemv_rejects_a_misaligned_k() {
        let blocks = vec![BlockTQ2_0_g128 {
            qs: [0u8; 32],
            d: half::f16::from_f32(1.0),
        }];
        let input = vec![0.0f32; 100];
        let mut out = vec![0.0f32; 1];
        assert!(gemv_two_bit_int8(Int8Tier::Scalar, &blocks, &input, &mut out, 1, 100).is_err());
    }

    #[test]
    fn gemv_reports_a_short_block_slice_by_name() {
        let blocks: Vec<BlockPQ2_0> = Vec::new();
        let input = vec![0.0f32; 128];
        let mut out = vec![0.0f32; 1];
        let err = gemv_two_bit_int8(Int8Tier::Scalar, &blocks, &input, &mut out, 1, 128)
            .expect_err("must reject");
        assert_eq!(err.buffer_name(), Some("blocks"));
    }

    /// The 1-bit entry points report the same named errors as the 2-bit
    /// ones (and as the f32 drivers they stand in for).
    #[test]
    fn one_bit_entry_points_report_errors_by_name() {
        let blocks = q1_blocks(2, 0x0B17);
        let input = vec![0.5f32; 64];
        let mut out = vec![0.0f32; 2];
        let err = gemv_1bit_g128_int8(Int8Tier::Scalar, &blocks, &input, &mut out, 2, 128)
            .expect_err("short input");
        assert_eq!(err.buffer_name(), Some("input"));
        let input = vec![0.5f32; 128];
        let err = gemm_1bit_g128_int8(Int8Tier::Scalar, &blocks, &input, &mut out, 1, 3, 128)
            .expect_err("short output");
        assert_eq!(err.buffer_name(), Some("output"));
        let mut out = vec![0.0f32; 3];
        let err = gemm_1bit_g128_int8(Int8Tier::Scalar, &blocks, &input, &mut out, 1, 3, 128)
            .expect_err("short blocks");
        assert_eq!(err.buffer_name(), Some("blocks"));
    }

    /// A NaN in the activation must reach the GEMV's output as NaN
    /// (matching the f32 reference path's NaN contagion through a dot
    /// product), never silently become a finite number because the
    /// poisoned block's contribution vanished.
    #[test]
    fn a_nan_activation_element_makes_the_gemv_output_nan() {
        let (n_rows, k) = (3usize, 128usize);
        let blocks: Vec<BlockPQ2_0> = (0..n_rows)
            .map(|_| BlockPQ2_0 {
                d: half::f16::from_f32(1.0),
                qs: [0b0110_0110u8; 32], // a mix of codes, never all-zero
            })
            .collect();
        let mut input = vec![0.5f32; k];
        input[3] = f32::NAN;
        let mut out = vec![0.0f32; n_rows];
        gemv_two_bit_int8(Int8Tier::Scalar, &blocks, &input, &mut out, n_rows, k)
            .expect("gemv with a NaN activation element");
        assert!(
            out.iter().all(|v| v.is_nan()),
            "a NaN activation element must poison every output row, got {out:?}"
        );

        let q1 = q1_blocks(n_rows, 0x0B18);
        let mut out = vec![0.0f32; n_rows];
        gemv_1bit_g128_int8(Int8Tier::Scalar, &q1, &input, &mut out, n_rows, k)
            .expect("1-bit gemv with a NaN activation element");
        assert!(out.iter().all(|v| v.is_nan()), "1-bit: got {out:?}");
    }

    /// `gemm_two_bit_int8` at `m == 1` must be bit-for-bit identical to
    /// [`gemv_two_bit_int8`] — not merely close, since integer accumulation
    /// is exact — with `n_rows` above [`INT8_PAR_MIN_ROWS`] so the Rayon
    /// split runs.
    #[test]
    fn gemm_at_m1_is_bit_identical_to_gemv() {
        let (n_rows, k) = (300usize, 2 * QK_PQ2_0);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x5EED_0099);
        let input = activations(k, 0x5EED_0098);

        for tier in supported_tiers() {
            let mut via_gemv = vec![0.0f32; n_rows];
            gemv_two_bit_int8(tier, &blocks, &input, &mut via_gemv, n_rows, k)
                .expect("gemv_two_bit_int8");
            let mut via_gemm = vec![0.0f32; n_rows];
            gemm_two_bit_int8(tier, &blocks, &input, &mut via_gemm, 1, n_rows, k)
                .expect("gemm_two_bit_int8 m=1");
            assert_bits_eq(&via_gemv, &via_gemm, &format!("{tier}: gemm(m=1) vs gemv"));
        }
    }

    /// Companion: `m == 2` must match a sequential per-row
    /// [`gemv_two_bit_int8`] sweep bit for bit.
    #[test]
    fn gemm_at_m2_matches_the_gemv_sweep_across_the_batch_parallel_threshold() {
        let (m, n_rows, k) = (2usize, 37usize, 2 * QK_PQ2_0);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x5EED_00BB);
        let input = activations(m * k, 0x5EED_00BA);

        for tier in supported_tiers() {
            let mut expect = vec![0.0f32; m * n_rows];
            for mi in 0..m {
                gemv_two_bit_int8(
                    tier,
                    &blocks,
                    &input[mi * k..(mi + 1) * k],
                    &mut expect[mi * n_rows..(mi + 1) * n_rows],
                    n_rows,
                    k,
                )
                .expect("gemv sweep row");
            }
            let mut got = vec![0.0f32; m * n_rows];
            gemm_two_bit_int8(tier, &blocks, &input, &mut got, m, n_rows, k)
                .expect("gemm_two_bit_int8 m=2");
            assert_bits_eq(&expect, &got, &format!("{tier}: gemm(m=2) vs gemv sweep"));
        }
    }

    /// The split decision itself, on synthetic thread counts so every
    /// boundary is exercised whatever this host's pool size is.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn int8_gemm_split_boundaries() {
        let big = INT8_PAR_MIN_ROWS;
        for threads in [2usize, 4, 8, 16] {
            assert_eq!(int8_gemm_split(1, big, threads), Int8GemmSplit::PerRowGemv);
            if threads > 2 {
                assert_eq!(
                    int8_gemm_split(2, big, threads),
                    Int8GemmSplit::PerRowGemv,
                    "m=2 on a {threads}-thread pool must fan out over weight rows"
                );
            }
            assert_eq!(
                int8_gemm_split(threads - 1, big, threads),
                Int8GemmSplit::PerRowGemv,
                "threads-1 batch rows must fan out over weight rows"
            );
            assert_eq!(
                int8_gemm_split(threads, big, threads),
                Int8GemmSplit::BatchSlabs,
                "a full batch per thread splits over batch rows"
            );
            assert_eq!(
                int8_gemm_split(2, big - 1, threads),
                Int8GemmSplit::BatchSlabs,
                "too few weight rows to split: batch slabs instead"
            );
        }
        // A one-thread pool never fans out.
        assert_eq!(int8_gemm_split(2, big, 1), Int8GemmSplit::BatchSlabs);
    }

    /// Every GEMM split — per-row GEMV fan-out, batch slabs, the
    /// sequential small-batch path — must match a sequential per-row GEMV
    /// sweep bit for bit, at every boundary batch size, with weight-row
    /// counts on both sides of [`INT8_PAR_MIN_ROWS`], on every tier.
    #[test]
    fn two_bit_gemm_matches_the_gemv_sweep_at_every_split_boundary() {
        let k = 2 * QK_TQ2_0_G128;
        for n_rows in [37usize, 300] {
            let blocks = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0xB0DE_0001);
            for m in boundary_batches() {
                let input = activations(m * k, 0xB0DE_1000 + m as u32);
                for tier in supported_tiers() {
                    let mut expect = vec![0.0f32; m * n_rows];
                    for mi in 0..m {
                        gemv_two_bit_int8(
                            tier,
                            &blocks,
                            &input[mi * k..(mi + 1) * k],
                            &mut expect[mi * n_rows..(mi + 1) * n_rows],
                            n_rows,
                            k,
                        )
                        .expect("gemv sweep row");
                    }
                    let mut got = vec![0.0f32; m * n_rows];
                    gemm_two_bit_int8(tier, &blocks, &input, &mut got, m, n_rows, k)
                        .expect("gemm_two_bit_int8");
                    assert_bits_eq(
                        &expect,
                        &got,
                        &format!("{tier}: 2-bit gemm m={m} n_rows={n_rows} vs gemv sweep"),
                    );
                }
            }
        }
    }

    /// The 1-bit GEMV's Rayon split (`n_rows = 300` crosses
    /// [`INT8_PAR_MIN_ROWS`]) must equal the same rows computed as uneven
    /// sequential sub-calls on the matching block slices, on every tier.
    #[test]
    fn one_bit_gemv_rayon_split_matches_sequential_sub_calls() {
        let (n_rows, k) = (300usize, 3 * QK1_0_G128);
        let blocks_per_row = k / QK1_0_G128;
        let blocks = q1_blocks(n_rows * blocks_per_row, 0x0B17_0300);
        let input = activations(k, 0x0B17_0301);
        for tier in supported_tiers() {
            let mut whole = vec![0.0f32; n_rows];
            gemv_1bit_g128_int8(tier, &blocks, &input, &mut whole, n_rows, k)
                .expect("whole 300-row gemv");
            let mut sequential = vec![0.0f32; n_rows];
            let mut row_start = 0usize;
            for rows in [113usize, 90, 97] {
                gemv_1bit_g128_int8(
                    tier,
                    &blocks[row_start * blocks_per_row..(row_start + rows) * blocks_per_row],
                    &input,
                    &mut sequential[row_start..row_start + rows],
                    rows,
                    k,
                )
                .expect("sequential sub-call");
                row_start += rows;
            }
            assert_eq!(row_start, n_rows);
            assert_bits_eq(
                &whole,
                &sequential,
                &format!("{tier}: 1-bit gemv split vs sequential sub-calls"),
            );
        }
    }

    /// The 1-bit GEMM twin of
    /// [`two_bit_gemm_matches_the_gemv_sweep_at_every_split_boundary`].
    #[test]
    fn one_bit_gemm_matches_the_gemv_sweep_at_every_split_boundary() {
        let k = 2 * QK1_0_G128;
        for n_rows in [23usize, 300] {
            let blocks = q1_blocks(n_rows * (k / QK1_0_G128), 0x0B17_1000);
            for m in boundary_batches() {
                let input = activations(m * k, 0x0B17_2000 + m as u32);
                for tier in supported_tiers() {
                    let mut expect = vec![0.0f32; m * n_rows];
                    for mi in 0..m {
                        gemv_1bit_g128_int8(
                            tier,
                            &blocks,
                            &input[mi * k..(mi + 1) * k],
                            &mut expect[mi * n_rows..(mi + 1) * n_rows],
                            n_rows,
                            k,
                        )
                        .expect("gemv sweep row");
                    }
                    let mut got = vec![0.0f32; m * n_rows];
                    gemm_1bit_g128_int8(tier, &blocks, &input, &mut got, m, n_rows, k)
                        .expect("gemm_1bit_g128_int8");
                    assert_bits_eq(
                        &expect,
                        &got,
                        &format!("{tier}: 1-bit gemm m={m} n_rows={n_rows} vs gemv sweep"),
                    );
                }
            }
        }
    }

    #[test]
    fn zero_rows_is_a_no_op() {
        let blocks: Vec<BlockPQ2_0> = Vec::new();
        let input = vec![1.0f32; 128];
        let mut out: Vec<f32> = Vec::new();
        gemv_two_bit_int8(Int8Tier::Scalar, &blocks, &input, &mut out, 0, 128)
            .expect("zero rows must succeed");
        assert!(out.is_empty());
        let q1: Vec<BlockQ1_0G128> = Vec::new();
        gemm_1bit_g128_int8(Int8Tier::Scalar, &q1, &input, &mut out, 1, 0, 128)
            .expect("zero weight rows must succeed");
    }

    /// A per-row GEMV sweep — the reference every GEMM path must reproduce
    /// bit for bit.
    #[cfg(target_arch = "aarch64")]
    fn gemv_sweep<B: Int8TwoBitBlock + Sync>(
        tier: Int8Tier,
        blocks: &[B],
        input: &[f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n_rows];
        for mi in 0..m {
            gemv_two_bit_int8(
                tier,
                blocks,
                &input[mi * k..(mi + 1) * k],
                &mut out[mi * n_rows..(mi + 1) * n_rows],
                n_rows,
                k,
            )
            .expect("gemv sweep row");
        }
        out
    }

    /// `gemm_two_bit_i8mm` called directly (not through the tier routing)
    /// on the `5 x 7` `PQ2_0` shape and data it was first pinned on, against
    /// the per-row GEMV reference every tier agrees on, bit for bit.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn i8mm_tile_matches_the_gemv_per_row_reference() {
        if !Int8Tier::NeonI8mm.is_supported() {
            eprintln!("skip: this host has no i8mm+dotprod");
            return;
        }
        let (m, n_rows, k) = (5usize, 7usize, 2 * QK_PQ2_0);
        let blocks_per_row = k / QK_PQ2_0;
        let mut lcg = 0x9EED_00AAu32 | 1;
        let mut next_u8 = move || {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (lcg >> 19) as u8
        };
        let blocks: Vec<BlockPQ2_0> = (0..n_rows * blocks_per_row)
            .map(|_| {
                let mut qs = [0u8; 32];
                for b in &mut qs {
                    *b = next_u8();
                }
                BlockPQ2_0 {
                    d: half::f16::from_f32(0.0625 + (next_u8() % 16) as f32 / 256.0),
                    qs,
                }
            })
            .collect();
        let input: Vec<f32> = (0..m * k)
            .map(|_| (i32::from(next_u8()) - 128) as f32 / 64.0)
            .collect();

        let expect = gemv_sweep(Int8Tier::NeonI8mm, &blocks, &input, m, n_rows, k);
        let act = Int8Activation::quantize(&input, m, k, BlockPQ2_0::QK, Int8Layout::Stride4)
            .expect("quantize");
        let lut16 = biased_lut16(&BlockPQ2_0::BIASED_LUT);
        let mut got = vec![0.0f32; m * n_rows];
        gemm_two_bit_i8mm(&blocks, &act, &mut got, m, n_rows, blocks_per_row, &lut16);
        assert_bits_eq(&expect, &got, "i8mm tile vs the gemv-per-row reference");
    }

    /// The decode-reuse `SMMLA` GEMM against the per-row GEMV sweep, bit
    /// for bit: odd and even `m` (the duplicated last batch row), odd and
    /// even `n_rows` (the duplicated last weight row), weight-row counts
    /// that span several Rayon tasks with a ragged last one, every format
    /// (`QK = 128`: `TQ2_0_g128` with its `0b11 -> 0` table, `PQ2_0` and
    /// `Q1_0_g128`; `QK = 64`: group-64 `Q2_0`) — including the `5 x 7`
    /// shape the per-block 2x2 tile was first pinned on.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn i8mm_gemm_matches_the_gemv_sweep_at_every_shape() {
        use oxibonsai_core::{BlockQ2_0G64, QK_Q2_0_G64};

        if !Int8Tier::NeonI8mm.is_supported() {
            eprintln!("skip: this host has no i8mm+dotprod");
            return;
        }
        let tier = Int8Tier::NeonI8mm;
        for (m, n_rows) in [
            (2usize, 7usize),
            (3, 8),
            (5, 7),
            (5, 37),
            (8, 300),
            (13, 1031),
        ] {
            let k = 3 * QK_TQ2_0_G128;
            let input = activations(m * k, 0x3A00 + m as u32);

            let tq2 = tq2_blocks(n_rows * (k / QK_TQ2_0_G128), 0x3A10 + n_rows as u32);
            let mut got = vec![0.0f32; m * n_rows];
            gemm_two_bit_int8(tier, &tq2, &input, &mut got, m, n_rows, k).expect("tq2 gemm");
            assert_bits_eq(
                &gemv_sweep(tier, &tq2, &input, m, n_rows, k),
                &got,
                &format!("TQ2_0_g128 i8mm gemm m={m} n_rows={n_rows}"),
            );

            let pq2 = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x3A20 + n_rows as u32);
            let mut got = vec![0.0f32; m * n_rows];
            gemm_two_bit_int8(tier, &pq2, &input, &mut got, m, n_rows, k).expect("pq2 gemm");
            assert_bits_eq(
                &gemv_sweep(tier, &pq2, &input, m, n_rows, k),
                &got,
                &format!("PQ2_0 i8mm gemm m={m} n_rows={n_rows}"),
            );

            let k64 = 5 * QK_Q2_0_G64;
            let input64 = activations(m * k64, 0x3A30 + m as u32);
            let mut rng = Lcg::new(0x3A40 + n_rows as u32);
            let g64: Vec<BlockQ2_0G64> = (0..n_rows * (k64 / QK_Q2_0_G64))
                .map(|_| {
                    let mut qs = [0u8; QK_Q2_0_G64 / 4];
                    for b in &mut qs {
                        *b = rng.next_u8();
                    }
                    BlockQ2_0G64 {
                        d: half::f16::from_f32(0.0625 + (rng.next_u8() % 16) as f32 / 256.0),
                        qs,
                    }
                })
                .collect();
            let mut got = vec![0.0f32; m * n_rows];
            gemm_two_bit_int8(tier, &g64, &input64, &mut got, m, n_rows, k64)
                .expect("q2_0_g64 gemm");
            assert_bits_eq(
                &gemv_sweep(tier, &g64, &input64, m, n_rows, k64),
                &got,
                &format!("Q2_0_g64 i8mm gemm m={m} n_rows={n_rows}"),
            );

            let q1 = q1_blocks(n_rows * (k / QK1_0_G128), 0x3A60 + n_rows as u32);
            let mut expect = vec![0.0f32; m * n_rows];
            for mi in 0..m {
                gemv_1bit_g128_int8(
                    tier,
                    &q1,
                    &input[mi * k..(mi + 1) * k],
                    &mut expect[mi * n_rows..(mi + 1) * n_rows],
                    n_rows,
                    k,
                )
                .expect("1-bit gemv sweep row");
            }
            let mut got = vec![0.0f32; m * n_rows];
            gemm_1bit_g128_int8(tier, &q1, &input, &mut got, m, n_rows, k).expect("q1 gemm");
            assert_bits_eq(
                &expect,
                &got,
                &format!("Q1_0_g128 i8mm gemm m={m} n_rows={n_rows}"),
            );
        }
    }

    /// NaN contagion survives the `SMMLA` path: a NaN activation element
    /// poisons exactly its own batch row's outputs.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn i8mm_gemm_propagates_a_nan_activation_to_its_own_batch_row_only() {
        if !Int8Tier::NeonI8mm.is_supported() {
            eprintln!("skip: this host has no i8mm+dotprod");
            return;
        }
        let (m, n_rows, k) = (3usize, 5usize, QK_TQ2_0_G128);
        let blocks = tq2_blocks(n_rows, 0x3A50);
        let q1 = q1_blocks(n_rows, 0x3A52);
        let mut input = activations(m * k, 0x3A51);
        input[k + 7] = f32::NAN; // batch row 1 only
        let mut out = vec![0.0f32; m * n_rows];
        gemm_two_bit_int8(Int8Tier::NeonI8mm, &blocks, &input, &mut out, m, n_rows, k)
            .expect("i8mm gemm");
        let mut out_q1 = vec![0.0f32; m * n_rows];
        gemm_1bit_g128_int8(Int8Tier::NeonI8mm, &q1, &input, &mut out_q1, m, n_rows, k)
            .expect("1-bit i8mm gemm");
        for result in [&out, &out_q1] {
            for (mi, row) in result.chunks(n_rows).enumerate() {
                if mi == 1 {
                    assert!(row.iter().all(|v| v.is_nan()), "row 1: {row:?}");
                } else {
                    assert!(row.iter().all(|v| v.is_finite()), "row {mi}: {row:?}");
                }
            }
        }
    }
}
