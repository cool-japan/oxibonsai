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
//! # Tiers
//!
//! | [`Int8Tier`] | `OXIBONSAI_KERNEL_TIER` | Requires | Inner loop |
//! |---|---|---|---|
//! | [`Int8Tier::Scalar`] | `int8-scalar` | — | `i32` MAC |
//! | [`Int8Tier::Neon`] | `neon-int8` | AArch64 | `SMULL` + `SADALP` |
//! | [`Int8Tier::NeonDot`] | `neon-dot` | `dotprod` | `SDOT` (16 MACs) |
//! | [`Int8Tier::NeonI8mm`] | `neon-i8mm` | `i8mm` + `dotprod` | `SDOT` (GEMV and GEMM — see the variant's doc) |
//! | [`Int8Tier::Avx512Vnni`] | `avx512-vnni` | `avx512f/bw/vnni` | `VPDPBUSD` (64 MACs) |
//!
//! Every tier computes the **same `i32`** per block — integer arithmetic is
//! exact and associative, so the only thing that changes between them is
//! how fast the sum is formed. The `f32` result is therefore identical
//! across tiers as well, which `int8_tier_parity` asserts bit-for-bit.
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
    /// ARMv8.6 `SMMLA` — 32 int8 MACs per instruction, for the 2x2 GEMM
    /// micro-kernel `gemm_two_bit_i8mm` implements.
    ///
    /// K-INT8 wave-4b: `gemm_two_bit_int8` does **not** route through that
    /// micro-kernel any more — measured on an M3, it lost to the plain
    /// `SDOT` row loop at every `M >= 2` (e.g. 27B `ffn_up` M=64: 326ms
    /// `SMMLA` vs 217ms `SDOT`), because each `(batch-row, weight-row)` tile
    /// pair re-decodes both weight rows from scratch instead of reusing a
    /// decode across pairs. This tier's GEMV *and* GEMM both run on `SDOT`
    /// today (same as [`Self::NeonDot`]) until a future fix hoists that
    /// decode out of the tile loop; `gemm_two_bit_i8mm` is kept, unrouted,
    /// and directly tested for when that lands.
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
/// process (K-INT8 wave-4b, minor[6]) — see that function's doc comment.
/// Not `#[cfg(test)]`: this is a production robustness fix, not a
/// test-only concern.
static TIER_DEMOTION_WARNED: std::sync::Once = std::sync::Once::new();

/// Serializes every test in this crate's `--lib` unit-test binary that
/// mutates or depends on a clean [`KERNEL_TIER_ENV`].
///
/// `std::env::set_var`/`remove_var` are `unsafe fn` (edition 2024) precisely
/// because a concurrent `std::env::var` on *any* key can observe a torn
/// `environ` while another thread mutates it — this is a whole-process
/// hazard, not a same-key one. Since `dispatch_prism.rs`'s `PrismKernel`
/// methods now call [`Int8Tier::from_env`] on every invocation (K-INT8
/// wave-4 fix-up), its bit-exact tests (`assert_eq!` on raw `f32` bits, no
/// tolerance) would go nondeterministically red if they ran concurrently
/// with this file's env-mutating test. Every test in this crate that reads
/// or writes [`KERNEL_TIER_ENV`] — here and in `dispatch_prism.rs` — takes
/// this lock first; nothing else in `oxibonsai-kernels` ever touches this
/// variable, so holding it for a test's short lifetime fully serializes the
/// hazard for this crate's test binary.
///
/// `#[cfg(test)]`, and referenced from `dispatch_prism.rs`'s own
/// `#[cfg(test)]` modules — this is a legal cross-module reference, not a
/// cross-crate one: `cargo test -p oxibonsai-kernels` rebuilds this whole
/// crate with `cfg(test)` active for the one `--lib` test binary, so both
/// modules see this item there. It adds zero surface to the non-test
/// build.
#[cfg(test)]
pub(crate) static KERNEL_TIER_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII guard around [`KERNEL_TIER_ENV_LOCK`] (K-INT8 wave-4b test-hygiene
/// fix-up): [`Self::acquire`] takes the lock, snapshots whatever
/// [`KERNEL_TIER_ENV`] currently holds — a developer's own shell export, or
/// nothing — and clears it, so a guarded test always starts from a
/// known-clean environment regardless of the ambient shell. [`Drop::drop`]
/// restores that exact snapshot, including when the guarded test body
/// panics: `Drop` still runs while unwinding, so a failed `assert!` between
/// a test's own `set_var` and its own cleanup `remove_var` no longer leaks
/// the mutated value to every later holder of the lock in this process (the
/// hazard the bare-`MutexGuard` version of this helper had).
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
        // moves into) serializes every reader/writer of `KERNEL_TIER_ENV`
        // in this crate's test binary — see `KERNEL_TIER_ENV_LOCK`'s doc
        // comment.
        unsafe {
            std::env::remove_var(KERNEL_TIER_ENV);
        }
        Self { _lock: lock, prior }
    }
}

#[cfg(test)]
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

    /// Parse a tier from its [`Self::name`] (case- and dash/underscore-
    /// insensitive). `None` for anything else.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let normalized = name.trim().to_ascii_lowercase().replace('_', "-");
        for tier in Self::ALL {
            if tier.name() == normalized {
                return Some(*tier);
            }
        }
        None
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
            // `i8mm` **and** `dotprod`: this tier's GEMV, and the odd
            // row/column tails of its `SMMLA` tiler, run on `SDOT`. Both are
            // mandatory from Armv8.6 (where `FEAT_I8MM` becomes mandatory,
            // and `FEAT_DotProd` has been since Armv8.4), but an Armv8.2
            // implementation may in principle carry `i8mm` alone — requiring
            // both here keeps every code path this tier can reach executable.
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

    /// Demote a tier this CPU cannot execute to the best one it can,
    /// warning once per call — the INT8 twin of
    /// `KernelDispatcher::clamp_tier_to_cpu` (K-03/sec-14), and what makes
    /// every safe entry point in this module safe on every host.
    #[must_use]
    pub fn clamp_to_cpu(self) -> Self {
        if self.is_supported() {
            return self;
        }
        let demoted = Self::best_available();
        // K-INT8 wave-4b (minor[6]): `dispatch_prism.rs`'s `PrismKernel`
        // methods call `Int8Tier::from_env` (and so `clamp_to_cpu`) on
        // *every* GEMV/GEMM call, by design — see that module's doc comment
        // for why the parsed tier itself is deliberately never cached (a
        // cached value would go stale mid-test while `dispatch_int8.rs`'s
        // env-mutating test is running). An unsupported-tier request would
        // therefore `tracing::warn!` on every one of those calls — hundreds
        // per token once decoding starts — so only the *warning* is
        // deduplicated (once per process), never the tier lookup itself.
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
    /// between machines still runs.
    #[must_use]
    pub fn from_env() -> Option<Self> {
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
/// [`two_bit_rows`] stays inside clippy's `too_many_arguments` budget.
#[derive(Debug, Clone, Copy)]
struct Int8Row<'a> {
    /// The row's int8 codes, in the kernel's expected layout.
    codes: &'a [i8],
    /// Per-block activation scales.
    scales: &'a [f32],
    /// Per-block exact code sums (the bias correction).
    sums: &'a [i32],
}

/// `output[row] = Σ_b d_b * s_b * (biased_dot_b − sum_b)` for one
/// activation row — the whole INT8 GEMV, minus the parallel split.
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

/// Rows per Rayon task for the INT8 GEMV — the same shape
/// `parallel.rs::rows_per_task` uses (K-16).
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn int8_rows_per_task(n_rows: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    (n_rows / (threads * 4)).clamp(8, 512)
}

/// Row count below which the INT8 GEMV stays on one thread.
#[cfg(not(target_arch = "wasm32"))]
const INT8_PAR_MIN_ROWS: usize = 256;

fn two_bit_gemv_int8<B: Int8TwoBitBlock + Sync>(
    tier: Int8Tier,
    blocks: &[B],
    act: &Int8Activation,
    output: &mut [f32],
    n_rows: usize,
    blocks_per_row: usize,
) -> KernelResult<()> {
    let lut16 = biased_lut16(&B::BIASED_LUT);
    let row = Int8Row {
        codes: act.codes_row(0),
        scales: act.scales_row(0),
        sums: act.sums_row(0),
    };

    // On WASM: no Rayon worker pool — stay sequential.
    #[cfg(target_arch = "wasm32")]
    {
        two_bit_rows(
            tier,
            blocks,
            row,
            &mut output[..n_rows],
            blocks_per_row,
            &lut16,
        );
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        if n_rows < INT8_PAR_MIN_ROWS {
            two_bit_rows(
                tier,
                blocks,
                row,
                &mut output[..n_rows],
                blocks_per_row,
                &lut16,
            );
            return Ok(());
        }
        let chunk = int8_rows_per_task(n_rows);
        output[..n_rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out_chunk)| {
                let row_start = ci * chunk;
                let rows = out_chunk.len();
                let slice =
                    &blocks[row_start * blocks_per_row..(row_start + rows) * blocks_per_row];
                two_bit_rows(tier, slice, row, out_chunk, blocks_per_row, &lut16);
            });
        Ok(())
    }
}

/// INT8 GEMV for any 2-bit format: `output[row] = dot(weight_row, input)`.
///
/// The activation is quantized **once** here and reused by all `n_rows`
/// weight rows (K-14, step 1), which is what makes the tier pay.
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
    two_bit_gemv_int8(tier, blocks, &act, output, n_rows, blocks_per_row)
}

/// Rows (batch dimension) per Rayon task for the plain INT8 GEMM — same
/// shape as [`int8_rows_per_task`], just chunking `m` instead of `n_rows`.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn int8_gemm_rows_per_task(m: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    m.div_ceil(threads).max(1).min(m)
}

/// Row count (batch dimension) below which the plain INT8 GEMM stays on one
/// thread — small enough that Rayon's task overhead is not worth it.
#[cfg(not(target_arch = "wasm32"))]
const INT8_GEMM_PAR_MIN_BATCH: usize = 2;

/// Row-parallel (batch dimension) plain INT8 GEMM: one [`two_bit_rows`] call
/// per output row, batch rows split across Rayon tasks. Bit-identical to a
/// sequential per-row loop — the split only decides *which* thread computes
/// a given row, never in what order that row's own terms are summed
/// (minor[1], K-INT8 wave-4b: `gemm_two_bit_int8` used to run this loop on
/// one thread regardless of `m`).
#[allow(clippy::too_many_arguments)]
fn two_bit_gemm_rows_par<B: Int8TwoBitBlock + Sync>(
    tier: Int8Tier,
    blocks: &[B],
    act: &Int8Activation,
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    blocks_per_row: usize,
    lut16: &[u8; 16],
) {
    #[cfg(target_arch = "wasm32")]
    {
        for mi in 0..m {
            let row = Int8Row {
                codes: act.codes_row(mi),
                scales: act.scales_row(mi),
                sums: act.sums_row(mi),
            };
            two_bit_rows(
                tier,
                blocks,
                row,
                &mut output[mi * n_rows..mi * n_rows + n_rows],
                blocks_per_row,
                lut16,
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        if m < INT8_GEMM_PAR_MIN_BATCH {
            for mi in 0..m {
                let row = Int8Row {
                    codes: act.codes_row(mi),
                    scales: act.scales_row(mi),
                    sums: act.sums_row(mi),
                };
                two_bit_rows(
                    tier,
                    blocks,
                    row,
                    &mut output[mi * n_rows..mi * n_rows + n_rows],
                    blocks_per_row,
                    lut16,
                );
            }
            return;
        }
        let chunk = int8_gemm_rows_per_task(m);
        output[..m * n_rows]
            .par_chunks_mut(chunk * n_rows)
            .enumerate()
            .for_each(|(ci, out_chunk)| {
                let m0 = ci * chunk;
                let rows = out_chunk.len() / n_rows;
                for r in 0..rows {
                    let mi = m0 + r;
                    let row = Int8Row {
                        codes: act.codes_row(mi),
                        scales: act.scales_row(mi),
                        sums: act.sums_row(mi),
                    };
                    two_bit_rows(
                        tier,
                        blocks,
                        row,
                        &mut out_chunk[r * n_rows..(r + 1) * n_rows],
                        blocks_per_row,
                        lut16,
                    );
                }
            });
    }
}

/// INT8 GEMM for any 2-bit format: `output[m, n] = dot(weight_n, input_m)`.
///
/// The whole `m x k` activation is quantized once, then every batch row is
/// evaluated against the shared weight matrix — `m == 1` delegates to
/// [`two_bit_gemv_int8`] (its own `n_rows`-parallel Rayon fan-out, the
/// dimension that is actually large at decode) and `m > 1` parallelizes
/// over the batch dimension via [`two_bit_gemm_rows_par`] (minor[0]/[1],
/// K-INT8 wave-4b: this whole function used to be single-threaded
/// regardless of `m`).
///
/// Does **not** use [`Int8Tier::NeonI8mm`]'s `SMMLA` 2x2 tile
/// (`gemm_two_bit_i8mm`) — see that variant's doc comment for why (minor[2],
/// same fix-up): measured slower than the plain `SDOT` row loop at every
/// `M >= 2` on an M3, so `NeonI8mm`'s GEMM runs on `SDOT` like
/// [`Int8Tier::NeonDot`] until a decode-reuse fix makes the tile actually
/// win.
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

    if m == 1 {
        return two_bit_gemv_int8(
            tier,
            blocks,
            &act,
            &mut output[..n_rows],
            n_rows,
            blocks_per_row,
        );
    }

    let lut16 = biased_lut16(&B::BIASED_LUT);
    two_bit_gemm_rows_par(
        tier,
        blocks,
        &act,
        output,
        m,
        n_rows,
        blocks_per_row,
        &lut16,
    );
    Ok(())
}

/// `SMMLA` 2x2 GEMM: two batch rows against two weight rows per tile, with
/// scalar-tier rows for the odd tails.
///
/// Every tile's `i32` is the same integer the other tiers form, so the
/// `f32` results are identical — `unrouted_i8mm_tile_still_matches_the_gemv_per_row_reference`
/// pins that directly, and it used to be exercised implicitly through
/// `gemm_two_bit_int8` before minor[2] (K-INT8 wave-4b) stopped routing
/// `Int8Tier::NeonI8mm`'s GEMM here (see that variant's doc comment: this
/// tile measured slower than the plain `SDOT` row loop on an M3). Kept,
/// `#[allow(dead_code)]`, for a future fix that hoists the weight-block
/// decode out of the `(mp, np)` loop and re-enables it.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code, clippy::too_many_arguments)]
fn gemm_two_bit_i8mm<B: Int8TwoBitBlock>(
    blocks: &[B],
    act: &Int8Activation,
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    blocks_per_row: usize,
    lut16: &[u8; 16],
) {
    let m_pairs = m / 2;
    let n_pairs = n_rows / 2;
    for mp in 0..m_pairs {
        let (m0, m1) = (mp * 2, mp * 2 + 1);
        let (codes0, codes1) = (act.codes_row(m0), act.codes_row(m1));
        let (scales0, scales1) = (act.scales_row(m0), act.scales_row(m1));
        let (sums0, sums1) = (act.sums_row(m0), act.sums_row(m1));
        for np in 0..n_pairs {
            let (n0, n1) = (np * 2, np * 2 + 1);
            let row0 = &blocks[n0 * blocks_per_row..(n0 + 1) * blocks_per_row];
            let row1 = &blocks[n1 * blocks_per_row..(n1 + 1) * blocks_per_row];
            let mut acc = [0.0f32; 4]; // [n0m0, n0m1, n1m0, n1m1]
            for bi in 0..blocks_per_row {
                let a0 = &codes0[bi * B::QK..(bi + 1) * B::QK];
                let a1 = &codes1[bi * B::QK..(bi + 1) * B::QK];
                // SAFETY: this function is only called for
                // `Int8Tier::NeonI8mm`, which `is_supported()` gates on
                // `is_aarch64_feature_detected!("i8mm")`.
                let tile = unsafe {
                    let lv = crate::simd_dot_int8::load_lut(lut16);
                    crate::simd_dot_int8::block_dot_two_bit_2x2_i8mm(
                        row0[bi].packed_codes(),
                        row1[bi].packed_codes(),
                        a0,
                        a1,
                        lv,
                    )
                };
                let d0 = row0[bi].block_scale();
                let d1 = row1[bi].block_scale();
                acc[0] += d0 * scales0[bi] * (tile[0] - sums0[bi]) as f32;
                acc[1] += d0 * scales1[bi] * (tile[1] - sums1[bi]) as f32;
                acc[2] += d1 * scales0[bi] * (tile[2] - sums0[bi]) as f32;
                acc[3] += d1 * scales1[bi] * (tile[3] - sums1[bi]) as f32;
            }
            output[m0 * n_rows + n0] = acc[0];
            output[m1 * n_rows + n0] = acc[1];
            output[m0 * n_rows + n1] = acc[2];
            output[m1 * n_rows + n1] = acc[3];
        }
        // Odd weight-row tail.
        for n in n_pairs * 2..n_rows {
            scalar_tail_cell(blocks, act, output, m0, n, n_rows, blocks_per_row, lut16);
            scalar_tail_cell(blocks, act, output, m1, n, n_rows, blocks_per_row, lut16);
        }
    }
    // Odd batch-row tail.
    for mi in m_pairs * 2..m {
        for n in 0..n_rows {
            scalar_tail_cell(blocks, act, output, mi, n, n_rows, blocks_per_row, lut16);
        }
    }
}

/// One `(batch row, weight row)` cell on the `SDOT` path — used for the
/// `SMMLA` tiler's odd tails. `#[allow(dead_code)]` for the same reason as
/// [`gemm_two_bit_i8mm`], its only caller.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code, clippy::too_many_arguments)]
fn scalar_tail_cell<B: Int8TwoBitBlock>(
    blocks: &[B],
    act: &Int8Activation,
    output: &mut [f32],
    mi: usize,
    ni: usize,
    n_rows: usize,
    blocks_per_row: usize,
    lut16: &[u8; 16],
) {
    let codes = act.codes_row(mi);
    let scales = act.scales_row(mi);
    let sums = act.sums_row(mi);
    let row = &blocks[ni * blocks_per_row..(ni + 1) * blocks_per_row];
    let mut sum = 0.0f32;
    for (bi, block) in row.iter().enumerate() {
        let a = &codes[bi * B::QK..(bi + 1) * B::QK];
        let acc = two_bit_block_dot(Int8Tier::NeonDot, block.packed_codes(), a, lut16) - sums[bi];
        sum += block.block_scale() * scales[bi] * acc as f32;
    }
    output[mi * n_rows + ni] = sum;
}

/// INT8 GEMV for the 1-bit `Q1_0_g128` format.
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
    one_bit_rows(
        tier,
        blocks,
        &act,
        0,
        &mut output[..n_rows],
        n_rows,
        blocks_per_row,
    );
    Ok(())
}

/// INT8 GEMM for the 1-bit `Q1_0_g128` format.
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
    for mi in 0..m {
        let (start, end) = (mi * n_rows, mi * n_rows + n_rows);
        one_bit_rows(
            tier,
            blocks,
            &act,
            mi,
            &mut output[start..end],
            n_rows,
            blocks_per_row,
        );
    }
    Ok(())
}

fn one_bit_rows(
    tier: Int8Tier,
    blocks: &[BlockQ1_0G128],
    act: &Int8Activation,
    row_index: usize,
    output: &mut [f32],
    n_rows: usize,
    blocks_per_row: usize,
) {
    let codes = act.codes_row(row_index);
    let scales = act.scales_row(row_index);
    let sums = act.sums_row(row_index);
    for (row, out) in output.iter_mut().enumerate().take(n_rows) {
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        let mut sum = 0.0f32;
        for (bi, block) in row_blocks.iter().enumerate() {
            let a = &codes[bi * QK1_0_G128..(bi + 1) * QK1_0_G128];
            let acc = one_bit_block_dot(tier, &block.qs, a) - sums[bi];
            sum += block.d.to_f32() * scales[bi] * acc as f32;
        }
        *out = sum;
    }
}

#[cfg(test)]
mod int8_dispatch_tests {
    use super::*;

    /// The guard the HARD CONSTRAINT asks for: with the environment clean,
    /// nothing selects this tier.
    ///
    /// Not `#[test]`-parallel-safe against a test that sets the variable, so
    /// the two live in one function; [`KERNEL_TIER_ENV_LOCK`] also
    /// serializes this against `dispatch_prism.rs`'s tests, which now
    /// observe this variable indirectly through every `PrismKernel` call.
    /// [`TierEnvGuard`] both takes that lock and guarantees a clean
    /// environment on entry and a restored one on exit (even on panic), so
    /// this test needs no manual `remove_var` of its own at either end.
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
        use oxibonsai_core::BlockTQ2_0_g128;
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
        use oxibonsai_core::BlockPQ2_0;
        let blocks: Vec<BlockPQ2_0> = Vec::new();
        let input = vec![0.0f32; 128];
        let mut out = vec![0.0f32; 1];
        let err = gemv_two_bit_int8(Int8Tier::Scalar, &blocks, &input, &mut out, 1, 128)
            .expect_err("must reject");
        assert_eq!(err.buffer_name(), Some("blocks"));
    }

    /// K-INT8 wave-4b (minor[6]): a NaN in the activation must reach the
    /// GEMV's output as NaN (matching the f32 reference path's NaN
    /// contagion through a dot product), never silently become a finite
    /// number because the poisoned block's contribution vanished.
    #[test]
    fn a_nan_activation_element_makes_the_gemv_output_nan() {
        use oxibonsai_core::BlockPQ2_0;
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
    }

    /// K-INT8 wave-4b minor[1]: `gemm_two_bit_int8` at `m == 1` must
    /// delegate to [`gemv_two_bit_int8`] and be bit-for-bit identical to it
    /// — not merely close, since integer accumulation is exact.
    #[test]
    fn gemm_at_m1_is_bit_identical_to_gemv() {
        use oxibonsai_core::BlockPQ2_0;
        let (n_rows, k) = (300usize, 2 * oxibonsai_core::QK_PQ2_0); // crosses INT8_PAR_MIN_ROWS
        let blocks_per_row = k / oxibonsai_core::QK_PQ2_0;
        let mut lcg = 0x5EED_0099u32 | 1;
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
        let input: Vec<f32> = (0..k)
            .map(|_| (i32::from(next_u8()) - 128) as f32 / 64.0)
            .collect();

        for tier in Int8Tier::ALL.iter().copied().filter(|t| t.is_supported()) {
            let mut via_gemv = vec![0.0f32; n_rows];
            gemv_two_bit_int8(tier, &blocks, &input, &mut via_gemv, n_rows, k)
                .expect("gemv_two_bit_int8");
            let mut via_gemm = vec![0.0f32; n_rows];
            gemm_two_bit_int8(tier, &blocks, &input, &mut via_gemm, 1, n_rows, k)
                .expect("gemm_two_bit_int8 m=1");
            for (i, (e, g)) in via_gemv.iter().zip(via_gemm.iter()).enumerate() {
                assert_eq!(
                    e.to_bits(),
                    g.to_bits(),
                    "{tier}: gemm(m=1) diverged from gemv at row {i}: {e} vs {g}"
                );
            }
        }
    }

    /// Companion: `m == 2` must go through [`two_bit_gemm_rows_par`]'s
    /// Rayon split (crossing [`INT8_GEMM_PAR_MIN_BATCH`]) and still match a
    /// sequential per-row [`gemv_two_bit_int8`] sweep bit for bit.
    #[test]
    fn gemm_at_m2_matches_the_gemv_sweep_across_the_batch_parallel_threshold() {
        use oxibonsai_core::BlockPQ2_0;
        let (m, n_rows, k) = (2usize, 37usize, 2 * oxibonsai_core::QK_PQ2_0);
        let blocks_per_row = k / oxibonsai_core::QK_PQ2_0;
        let mut lcg = 0x5EED_00BBu32 | 1;
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

        for tier in Int8Tier::ALL.iter().copied().filter(|t| t.is_supported()) {
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
            for (i, (e, g)) in expect.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    e.to_bits(),
                    g.to_bits(),
                    "{tier}: gemm(m=2) diverged from the gemv sweep at cell {i}: {e} vs {g}"
                );
            }
        }
    }

    #[test]
    fn zero_rows_is_a_no_op() {
        use oxibonsai_core::BlockPQ2_0;
        let blocks: Vec<BlockPQ2_0> = Vec::new();
        let input = vec![1.0f32; 128];
        let mut out: Vec<f32> = Vec::new();
        gemv_two_bit_int8(Int8Tier::Scalar, &blocks, &input, &mut out, 0, 128)
            .expect("zero rows must succeed");
        assert!(out.is_empty());
    }

    /// K-INT8 wave-4b minor[2]: `gemm_two_bit_i8mm` (the `SMMLA` 2x2 tile)
    /// is deliberately not routed to from `gemm_two_bit_int8` any more —
    /// see [`Int8Tier::NeonI8mm`]'s doc comment for the measured numbers.
    /// It is kept, `#[allow(dead_code)]`, for a future decode-reuse fix;
    /// this test is what keeps it alive and proven bit-for-bit correct in
    /// the meantime — without it, the tile would be genuine dead code,
    /// unreachable from anywhere in the crate.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn unrouted_i8mm_tile_still_matches_the_gemv_per_row_reference() {
        use oxibonsai_core::BlockPQ2_0;

        if !Int8Tier::NeonI8mm.is_supported() {
            eprintln!("skip: this host has no i8mm+dotprod");
            return;
        }

        let (m, n_rows, k) = (5usize, 7usize, 2 * oxibonsai_core::QK_PQ2_0);
        let blocks_per_row = k / oxibonsai_core::QK_PQ2_0;
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
        let mut input = vec![0.0f32; m * k];
        for x in &mut input {
            *x = (i32::from(next_u8()) - 128) as f32 / 64.0;
        }

        // Reference: the exact same per-row GEMV every tier (including
        // `NeonI8mm`, which runs `SDOT` for GEMV) already agrees on, per
        // `every_int8_tier_produces_bit_identical_output` in
        // `int8_tier_parity.rs`.
        let mut expect = vec![0.0f32; m * n_rows];
        for mi in 0..m {
            gemv_two_bit_int8(
                Int8Tier::NeonI8mm,
                &blocks,
                &input[mi * k..(mi + 1) * k],
                &mut expect[mi * n_rows..(mi + 1) * n_rows],
                n_rows,
                k,
            )
            .expect("gemv reference");
        }

        let act = Int8Activation::quantize(&input, m, k, BlockPQ2_0::QK, Int8Layout::Stride4)
            .expect("quantize");
        let lut16 = biased_lut16(&BlockPQ2_0::BIASED_LUT);
        let mut got = vec![0.0f32; m * n_rows];
        gemm_two_bit_i8mm(&blocks, &act, &mut got, m, n_rows, blocks_per_row, &lut16);

        for (i, (e, g)) in expect.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                e.to_bits(),
                g.to_bits(),
                "unrouted i8mm tile diverged from the gemv-per-row reference \
                 at cell {i}: {e} vs {g}"
            );
        }
    }
}
