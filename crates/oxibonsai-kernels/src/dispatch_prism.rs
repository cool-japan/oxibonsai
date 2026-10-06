//! [`PrismKernel`] dispatch (design doc §2.7): the three new PrismML
//! Bonsai 2 quant-format GEMV/GEMM kernels (`PQ2_0`, `PTQ1_0`, mainline
//! group-64 `Q2_0`) plus the Gated-DeltaNet/Hadamard/SSM hybrid-math
//! primitives.
//!
//! Split into its own sibling file for the same reason as
//! `dispatch_std_quant.rs`: a single `impl Trait for Type` cannot itself be
//! split, and keeping every `PrismKernel` method together (rather than
//! stuffing 16 more methods into `dispatch.rs`, which is already at its
//! 2000-line budget) keeps each dispatch file a coherent, reviewable unit.
//!
//! ## GEMV/GEMM tier dispatch
//!
//! `PQ2_0`/`PTQ1_0`/group-64 `Q2_0` each have a real per-tier kernel
//! (scalar reference, NEON, AVX2 — the acceptance criteria only required
//! NEON parity, so there is no dedicated AVX-512 kernel; `KernelTier::Avx512`
//! therefore falls through to the AVX2 kernel, which every real AVX-512F CPU
//! also supports). The Metal and CUDA kernels for these three formats exist,
//! but inside the hybrid (`qwen35`) runner rather than behind
//! [`KernelDispatcher`]: the Metal sources are
//! `gpu_backend/kernel_sources/qwen35.rs` and `qwen35_gemm.rs` (driven by
//! `gpu_backend/metal_full_layer/qwen35*.rs`) and the CUDA ones
//! `gpu_backend/kernel_sources/cuda_qwen35_kernels.rs`. The runner decodes whole
//! layers on the device, so no per-call GEMV or GEMM entry point of this
//! dispatcher reaches them, and a `KernelTier::Gpu` dispatcher therefore
//! routes these formats straight to the best *CPU* tier — the same K-17 policy
//! `dispatch_std_quant.rs` and `dispatch.rs`'s ternary/1-bit arms use, so a
//! GPU-tier dispatcher never regresses to the scalar kernel for a format that
//! has no per-call GPU kernel.
//!
//! That one mapping now lives in `KernelDispatcher::prism_tier` instead of
//! six near-identical `cpu_*_fallback` methods: the six differed
//! only in which kernel they called, and only the three GEMV ones recorded
//! the fallback tier for the K-17 regression tests. The three GEMM sites
//! now record it too, which a single shared mapping gives for free and
//! cannot drift out of again.
//!
//! ## Tiling and row/batch parallelism
//!
//! Before the register-blocked path every Prism GEMM was a literal loop of GEMVs and every entry
//! point was single-threaded: `rg rayon` over the four Prism kernel files
//! returned nothing, so the chunked CPU prefill would re-stream the full
//! 7.2 GB `PQ2_0` weight set once per prompt token on one core. Both halves
//! are fixed here, behind the unchanged public entry points:
//!
//! - **Register blocking**: `gemm_*` routes to the `*_blocked` kernels
//!   (`dequant_prism`, `gemv_ptq1`, `simd_prism_neon`, `simd_prism_avx2`),
//!   which decode a weight block once per [`PRISM_GEMM_MR`] batch rows.
//! - **Rayon**: `prism_gemm_par` splits the batch dimension into slabs of
//!   whole batch rows (so the register blocking survives the split) and
//!   `prism_gemv_par` splits the weight-row dimension into chunks, both
//!   above the platform-tuned thresholds [`crate::tuning`] already computes
//!   for the ternary/1-bit drivers.
//!
//! Both are **bit-exact**: a register block keeps each (batch row, weight
//! row) pair's multiply-add sequence and per-block `sum += d * acc` order
//! exactly as the GEMV sweep had it, and a Rayon split only decides *which
//! thread* evaluates an output element, never in what order its terms are
//! summed. `prism_blocked_tests` asserts both, per format and per tier, with
//! `assert_eq!` on the raw `f32` bits — never a tolerance.
//!
//! ## The rest (`fwht_*`, `gdn_*`, `conv1d_*`, `l2_norm`, `rms_norm_gated`,
//! `sigmoid_mul`, `softplus`, `rope_partial_splithalf`)
//!
//! See [`crate::traits::PrismKernel`]'s doc comment: each of these already
//! self-dispatches NEON/AVX2/scalar internally, so every tier converges on
//! the one free function.
//!
//! ## INT8 tier wiring (K-14)
//!
//! `LinearPQ2_0::forward`/`forward_batch` and `LinearQ2_0G64::forward`/
//! `forward_batch` call `self.kernel.gemv_pq2_0(..)` / `.gemm_pq2_0(..)` /
//! `.gemv_q2_0_g64(..)` / `.gemm_q2_0_g64(..)` on their stored
//! `Arc<KernelDispatcher>`, and `impl PrismKernel for KernelDispatcher` is
//! those four methods: each reads [`Int8Tier::from_env`] first and, when
//! `OXIBONSAI_KERNEL_TIER` names a tier, runs the matching `dispatch_int8`
//! kernel. The Prism formats have no GPU kernel, so this applies whichever
//! `KernelTier` the dispatcher is on.
//! `gemv_and_gemm_pq2_0_route_through_the_int8_tier_when_asked` and
//! `gemv_and_gemm_q2_0_g64_route_through_the_int8_tier_when_asked` pin it.
//!
//! Each call reads [`Int8Tier::from_env`] fresh — never cached — so a
//! change to the variable takes effect on the next call; one `env::var` per
//! GEMV/GEMM call (not per weight row) is not measurable next to the matmul
//! it gates. In this crate's unit-test build the variable is honoured only
//! on a thread holding `dispatch_int8::TierEnvGuard` (see
//! `dispatch_int8::KERNEL_TIER_ENV_LOCK`), so the bit-exact tests below
//! cannot observe another test's value.
//!
//! One format is deliberately **not** wired here:
//!
//! - `gemv_ptq1_0`/`gemm_ptq1_0` (`PTQ1_0`, ggml id 143): its 28-byte block
//!   packs five base-3 trits per byte (`qs[24]` + `qh[2]`), not the
//!   2-bit-per-weight codes [`crate::simd_dot_int8::Int8TwoBitBlock`]'s two
//!   LUT tables decode. No `Int8TwoBitBlock` impl exists for
//!   [`BlockPTQ1_0`] (only [`oxibonsai_core::BlockTQ2_0_g128`],
//!   [`BlockPQ2_0`] and [`BlockQ2_0G64`] at `simd_dot_int8.rs:524/537/550`
//!   do), and the INT8 dispatch is scoped to exactly those three formats
//!   — never a fourth for PTQ1_0. Building a
//!   base-3 int8 decode kernel is a real, separate undertaking, not a
//!   small change; `gemv_ptq1_0_ignores_the_int8_tier_env_var`
//!   pins this as the intended behaviour, not a gap.
//!
//! (The native ternary `TQ2_0_g128` and 1-bit `Q1_0_g128` formats are wired
//! too, in `dispatch.rs` — `OneBitKernel::{gemv, gemm}`,
//! `TernaryKernel::{gemv_ternary_g128, gemm_ternary_g128}` — and at the
//! entry of the `parallel` / `parallel_tiled` drivers; see
//! `KernelDispatcher::native_int8_tier`. Unlike the Prism formats they never
//! divert a `KernelTier::Gpu` dispatcher.)

use oxibonsai_core::{BlockPQ2_0, BlockPTQ1_0, BlockQ2_0G64, QK_PQ2_0, QK_PTQ1_0, QK_Q2_0_G64};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use crate::dequant_prism::PRISM_GEMM_MR;
use crate::dispatch::{KernelDispatcher, KernelTier};
use crate::dispatch_int8::{self, Int8Tier};
use crate::error::KernelResult;
use crate::gated_delta_net::GdnState;
use crate::traits::PrismKernel;
#[cfg(not(target_arch = "wasm32"))]
use crate::tuning::PlatformProfile;

// ─── Tier mapping ────────────────────────────────────────────────────────

/// Which per-tier Prism arithmetic a [`KernelTier`] maps onto.
///
/// Deliberately a *separate* enum from [`KernelTier`], mirroring
/// `gemm_ternary.rs::BlockedTier`: the Prism family has three kernel tiers,
/// not five, and collapsing `Avx512 -> Avx2` / `Gpu -> best CPU` once, here,
/// keeps every driver below a plain three-arm match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrismTier {
    /// Pure scalar reference (`dequant_prism` / `gemv_ptq1`).
    Scalar,
    /// NEON (`simd_prism_neon`).
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// AVX2 + FMA (`simd_prism_avx2`); also serves `KernelTier::Avx512`.
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

/// Map a [`KernelTier`] onto the Prism kernel tier that executes it.
///
/// `KernelTier::Gpu` must be resolved by the caller
/// ([`KernelDispatcher::prism_tier`]) before reaching here, because doing so
/// has an observable side effect (recording the fallback tier for the K-17
/// regression tests); it is mapped to [`PrismTier::Scalar`] defensively so
/// this function stays total.
fn prism_tier_of(tier: KernelTier) -> PrismTier {
    match tier {
        KernelTier::Reference => PrismTier::Scalar,
        #[cfg(target_arch = "aarch64")]
        KernelTier::Neon => PrismTier::Neon,
        #[cfg(target_arch = "x86_64")]
        KernelTier::Avx2 | KernelTier::Avx512 => PrismTier::Avx2,
        #[cfg(feature = "gpu")]
        KernelTier::Gpu => PrismTier::Scalar,
    }
}

impl KernelDispatcher {
    /// The Prism CPU tier this dispatcher's GEMV/GEMM calls execute on.
    ///
    /// A `KernelTier::Gpu` dispatcher resolves through
    /// [`KernelDispatcher::cpu_tier`] (K-17: no Metal/CUDA Prism kernel
    /// exists yet) and records the routing decision so
    /// `dispatch.rs::LAST_GPU_FALLBACK_TIER` sees it — for **every** Prism
    /// entry point, GEMV and GEMM alike (previously only the three GEMV
    /// fallbacks recorded).
    pub(crate) fn prism_tier(&self) -> PrismTier {
        match self.tier() {
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                let cpu = Self::cpu_tier();
                #[cfg(test)]
                crate::dispatch::record_gpu_fallback_tier(cpu);
                prism_tier_of(cpu)
            }
            other => prism_tier_of(other),
        }
    }
}

// ─── Per-tier kernel selection ───────────────────────────────────────────

/// A Prism GEMV kernel: `(blocks, input, output, n_rows, k)`.
type PrismGemvFn<B> = fn(&[B], &[f32], &mut [f32], usize, usize) -> KernelResult<()>;

/// A register-blocked Prism GEMM kernel: `(blocks, input, output, m, n_rows, k)`.
type PrismGemmFn<B> = fn(&[B], &[f32], &mut [f32], usize, usize, usize) -> KernelResult<()>;

/// SAFETY (shared by every `unsafe` block in the four selectors below): each
/// `PrismTier` variant is only constructible on the architecture whose
/// `#[target_feature]` the kernel requires — `Neon` only under
/// `cfg(target_arch = "aarch64")`, where NEON is the ISA baseline, and
/// `Avx2` only from a `KernelTier::Avx2`/`Avx512` that
/// `KernelDispatcher::clamp_tier_to_cpu` already re-validated with
/// `is_x86_feature_detected!`.
fn gemv_pq2_0_kernel(tier: PrismTier) -> PrismGemvFn<BlockPQ2_0> {
    match tier {
        PrismTier::Scalar => crate::dequant_prism::gemv_pq2_0,
        #[cfg(target_arch = "aarch64")]
        PrismTier::Neon => {
            |b, i, o, n, k| unsafe { crate::simd_prism_neon::gemv_pq2_0_neon(b, i, o, n, k) }
        }
        #[cfg(target_arch = "x86_64")]
        PrismTier::Avx2 => {
            |b, i, o, n, k| unsafe { crate::simd_prism_avx2::gemv_pq2_0_avx2(b, i, o, n, k) }
        }
    }
}

/// See [`gemv_pq2_0_kernel`] for the shared safety argument.
fn gemm_pq2_0_kernel(tier: PrismTier) -> PrismGemmFn<BlockPQ2_0> {
    match tier {
        PrismTier::Scalar => crate::dequant_prism::gemm_pq2_0_blocked,
        #[cfg(target_arch = "aarch64")]
        PrismTier::Neon => |b, i, o, m, n, k| unsafe {
            crate::simd_prism_neon::gemm_pq2_0_neon_blocked(b, i, o, m, n, k)
        },
        #[cfg(target_arch = "x86_64")]
        PrismTier::Avx2 => |b, i, o, m, n, k| unsafe {
            crate::simd_prism_avx2::gemm_pq2_0_avx2_blocked(b, i, o, m, n, k)
        },
    }
}

/// See [`gemv_pq2_0_kernel`] for the shared safety argument.
fn gemv_ptq1_0_kernel(tier: PrismTier) -> PrismGemvFn<BlockPTQ1_0> {
    match tier {
        PrismTier::Scalar => crate::gemv_ptq1::gemv_ptq1_0,
        #[cfg(target_arch = "aarch64")]
        PrismTier::Neon => {
            |b, i, o, n, k| unsafe { crate::simd_prism_neon::gemv_ptq1_0_neon(b, i, o, n, k) }
        }
        #[cfg(target_arch = "x86_64")]
        PrismTier::Avx2 => {
            |b, i, o, n, k| unsafe { crate::simd_prism_avx2::gemv_ptq1_0_avx2(b, i, o, n, k) }
        }
    }
}

/// See [`gemv_pq2_0_kernel`] for the shared safety argument.
fn gemm_ptq1_0_kernel(tier: PrismTier) -> PrismGemmFn<BlockPTQ1_0> {
    match tier {
        PrismTier::Scalar => crate::gemv_ptq1::gemm_ptq1_0_blocked,
        #[cfg(target_arch = "aarch64")]
        PrismTier::Neon => |b, i, o, m, n, k| unsafe {
            crate::simd_prism_neon::gemm_ptq1_0_neon_blocked(b, i, o, m, n, k)
        },
        #[cfg(target_arch = "x86_64")]
        PrismTier::Avx2 => |b, i, o, m, n, k| unsafe {
            crate::simd_prism_avx2::gemm_ptq1_0_avx2_blocked(b, i, o, m, n, k)
        },
    }
}

/// See [`gemv_pq2_0_kernel`] for the shared safety argument.
fn gemv_q2_0_g64_kernel(tier: PrismTier) -> PrismGemvFn<BlockQ2_0G64> {
    match tier {
        PrismTier::Scalar => crate::dequant_prism::gemv_q2_0_g64,
        #[cfg(target_arch = "aarch64")]
        PrismTier::Neon => {
            |b, i, o, n, k| unsafe { crate::simd_prism_neon::gemv_q2_0_g64_neon(b, i, o, n, k) }
        }
        #[cfg(target_arch = "x86_64")]
        PrismTier::Avx2 => {
            |b, i, o, n, k| unsafe { crate::simd_prism_avx2::gemv_q2_0_g64_avx2(b, i, o, n, k) }
        }
    }
}

/// See [`gemv_pq2_0_kernel`] for the shared safety argument.
fn gemm_q2_0_g64_kernel(tier: PrismTier) -> PrismGemmFn<BlockQ2_0G64> {
    match tier {
        PrismTier::Scalar => crate::dequant_prism::gemm_q2_0_g64_blocked,
        #[cfg(target_arch = "aarch64")]
        PrismTier::Neon => |b, i, o, m, n, k| unsafe {
            crate::simd_prism_neon::gemm_q2_0_g64_neon_blocked(b, i, o, m, n, k)
        },
        #[cfg(target_arch = "x86_64")]
        PrismTier::Avx2 => |b, i, o, m, n, k| unsafe {
            crate::simd_prism_avx2::gemm_q2_0_g64_avx2_blocked(b, i, o, m, n, k)
        },
    }
}

// ─── Parallel drivers ────────────────────────────────────────────────────

/// Weight rows per Rayon work-item for [`prism_gemv_par`].
///
/// Same shape as `parallel.rs::rows_per_task` (K-16): roughly four tasks per
/// worker so work-stealing can still balance, clamped to `[8, 512]` so a
/// small matrix still gets real parallelism and a huge one never serializes
/// on one oversized task.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn prism_rows_per_task(n_rows: usize) -> usize {
    let num_threads = rayon::current_num_threads().max(1);
    (n_rows / (num_threads * 4)).clamp(8, 512)
}

/// Batch rows per Rayon work-item for [`prism_gemm_par`].
///
/// One slab per worker thread, floored at [`PRISM_GEMM_MR`] — a slab shorter
/// than the register-block factor cannot fill the register block and would
/// throw the weight-decode reuse away, which is the entire point of the
/// blocked kernel. Same shape as `parallel.rs::gemm_batch_chunk_rows` (K-18).
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn prism_gemm_chunk_rows(m: usize, mr: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    m.div_ceil(threads).max(mr).min(m).max(1)
}

/// Row-parallel Prism GEMV driver.
///
/// Validation is hoisted out of the per-chunk kernel calls so a bad call
/// reports exactly the error (and buffer name) the sequential kernel would
/// have reported, instead of panicking while slicing `blocks`.
///
/// Bit-exact: every output row is still computed by one thread with the
/// kernel's own accumulation order; the split only chooses *which* thread.
fn prism_gemv_par<B: Sync>(
    blocks: &[B],
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    k: usize,
    qk: usize,
    kernel: PrismGemvFn<B>,
) -> KernelResult<()> {
    let blocks_per_row = crate::dequant_prism::validate_prism_gemm(
        blocks.len(),
        input.len(),
        output.len(),
        1,
        n_rows,
        k,
        qk,
    )?;
    if n_rows == 0 {
        return Ok(());
    }

    // On WASM: no Rayon worker pool — stay sequential.
    #[cfg(target_arch = "wasm32")]
    {
        let _ = blocks_per_row;
        kernel(blocks, input, output, n_rows, k)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        if n_rows < PlatformProfile::global_thresholds().par_gemv_min_rows {
            return kernel(blocks, input, output, n_rows, k);
        }
        let chunk_rows = prism_rows_per_task(n_rows);
        output[..n_rows]
            .par_chunks_mut(chunk_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_rows;
                let rows = out_chunk.len();
                let block_start = row_start * blocks_per_row;
                let block_end = (row_start + rows) * blocks_per_row;
                kernel(&blocks[block_start..block_end], input, out_chunk, rows, k)
            })?;
        Ok(())
    }
}

/// `m`-dispatching Prism GEMM entry: [`prism_gemm_chunk_rows`] floors the
/// parallel slab width at
/// [`PRISM_GEMM_MR`] (`chunk_rows = max(m.div_ceil(threads), MR)`), so the
/// batch-parallel path collapses to a **single** slab — `min(chunk_rows, m)
/// == m` — for every `m <= MR`, not only `m < MR`: at `m == MR` exactly,
/// `m.div_ceil(threads) <= MR` for any `threads >= 1`, so the `max` still
/// picks `MR == m` and one task covers the whole batch. Decode (`m == 1`)
/// always hit this, and so does every `m` up to and including `MR` (8 on
/// this host) even once `m` clears
/// [`crate::tuning::PlatformThresholds::par_gemm_min_batch`]. Measured on
/// this M3 at the 27B `ffn_up` shape (`[5120, 17408]`): `f32`
/// `gemm_pq2_0(m=1)` 26.7ms vs `gemv_pq2_0` 5.2ms (5.1x), and `m=5` 120ms —
/// *slower* than `m=64`'s 98.6ms despite doing 12x less work, because `m=64`
/// gets real 8-way slab parallelism and `m=5` gets none.
///
/// The fix loops [`prism_gemv_par`] once per batch row instead of taking the
/// single-threaded blocked path, which gets each row real
/// weight-row-parallel fan-out (`n_rows` — 17408 above — is normally far
/// larger than `PRISM_GEMM_MR`, so this is where the real parallelism is for
/// small `m`). This is **bit-exact**, not an approximation: a register
/// block's per-`(batch row, weight row)` accumulation order already matches
/// the plain GEMV sweep exactly (this module's doc comment, and
/// `*_blocked_gemm_is_bit_identical_to_the_gemv_sweep`'s `TAIL_M = 13`
/// case), so looping the GEMV per row reproduces the blocked kernel's bits
/// for every `m`, not only `m == 1`.
///
/// The shape is validated whole, before any row is sliced out of `input` /
/// `output`, so a mismatched buffer still reports the same named error the
/// blocked path would have (never a slice-index panic).
///
/// Note on `m` just above [`PRISM_GEMM_MR`] (e.g. 9..~63 on an 8-thread, MR=8
/// host): [`prism_gemm_chunk_rows`]'s own floor still caps the slab width at
/// `MR`, so parallelism there ramps up gradually with `m` (2 slabs at
/// `m=9..15`, not the full thread count) rather than jumping straight to
/// `m=64`'s full 8-way split. That is strictly better than the earlier
/// single-slab behavior this function replaces for `m <= MR`, and matches
/// the pre-existing, previously-accepted chunking this function defers to
/// for `m > MR` — narrowing it further is a distinct, separate
/// optimization.
#[allow(clippy::too_many_arguments)]
fn prism_gemm_dispatch<B: Sync>(
    blocks: &[B],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
    qk: usize,
    gemv_kernel: PrismGemvFn<B>,
    blocked: PrismGemmFn<B>,
) -> KernelResult<()> {
    crate::dequant_prism::validate_prism_gemm(
        blocks.len(),
        input.len(),
        output.len(),
        m,
        n_rows,
        k,
        qk,
    )?;
    if m == 0 || n_rows == 0 {
        return Ok(());
    }

    // On WASM there is no Rayon worker pool (`prism_gemv_par`'s own
    // WASM arm is already sequential, and so is `prism_gemm_par`'s), so
    // `PRISM_GEMM_MR` — a *parallel* slab-width floor — has nothing to do
    // with the choice there; always take the register-blocked path below,
    // which still gets weight-decode reuse across batch rows even
    // single-threaded. `gemv_kernel` genuinely goes unused on this arch.
    #[cfg(target_arch = "wasm32")]
    let _ = gemv_kernel;

    #[cfg(not(target_arch = "wasm32"))]
    if m <= PRISM_GEMM_MR {
        for mi in 0..m {
            prism_gemv_par(
                blocks,
                &input[mi * k..(mi + 1) * k],
                &mut output[mi * n_rows..(mi + 1) * n_rows],
                n_rows,
                k,
                qk,
                gemv_kernel,
            )?;
        }
        return Ok(());
    }

    prism_gemm_par(blocks, input, output, m, n_rows, k, qk, blocked)
}

/// Batch-parallel, register-blocked Prism GEMM driver.
///
/// Rayon splits the **batch** dimension into slabs of whole batch rows, so
/// each task runs a genuine `m > 1` register-blocked GEMM over the whole
/// weight matrix and the decoded-block reuse survives the split (K-18's
/// shape, ported to the Prism family).
///
/// Bit-exact, for the same reason as [`prism_gemv_par`].
#[allow(clippy::too_many_arguments)]
fn prism_gemm_par<B: Sync>(
    blocks: &[B],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n_rows: usize,
    k: usize,
    qk: usize,
    blocked: PrismGemmFn<B>,
) -> KernelResult<()> {
    crate::dequant_prism::validate_prism_gemm(
        blocks.len(),
        input.len(),
        output.len(),
        m,
        n_rows,
        k,
        qk,
    )?;
    if m == 0 || n_rows == 0 {
        return Ok(());
    }

    // On WASM: no Rayon worker pool — stay sequential (still blocked).
    #[cfg(target_arch = "wasm32")]
    {
        blocked(blocks, input, output, m, n_rows, k)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        if m < PlatformProfile::global_thresholds().par_gemm_min_batch {
            return blocked(blocks, input, output, m, n_rows, k);
        }
        let chunk_rows = prism_gemm_chunk_rows(m, PRISM_GEMM_MR);
        output[..m * n_rows]
            .par_chunks_mut(chunk_rows * n_rows)
            .enumerate()
            .try_for_each(|(chunk_idx, out_chunk)| {
                let m0 = chunk_idx * chunk_rows;
                let rows = out_chunk.len() / n_rows;
                let input_chunk = &input[m0 * k..(m0 + rows) * k];
                blocked(blocks, input_chunk, out_chunk, rows, n_rows, k)
            })?;
        Ok(())
    }
}

impl PrismKernel for KernelDispatcher {
    fn gemv_pq2_0(
        &self,
        blocks: &[BlockPQ2_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        // See this module's doc comment. `None` (the default, unset
        // environment) falls through to exactly today's call, unchanged,
        // so the default stays bit-identical.
        if let Some(tier) = Int8Tier::from_env() {
            return dispatch_int8::gemv_two_bit_int8(tier, blocks, input, output, n_rows, k);
        }
        let kernel = gemv_pq2_0_kernel(self.prism_tier());
        prism_gemv_par(blocks, input, output, n_rows, k, QK_PQ2_0, kernel)
    }

    fn gemm_pq2_0(
        &self,
        blocks: &[BlockPQ2_0],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        // See `gemv_pq2_0` above.
        if let Some(tier) = Int8Tier::from_env() {
            return dispatch_int8::gemm_two_bit_int8(tier, blocks, input, output, m, n_rows, k);
        }
        let tier = self.prism_tier();
        prism_gemm_dispatch(
            blocks,
            input,
            output,
            m,
            n_rows,
            k,
            QK_PQ2_0,
            gemv_pq2_0_kernel(tier),
            gemm_pq2_0_kernel(tier),
        )
    }

    /// **Not** wired to [`Int8Tier`]: see this module's doc comment ("One
    /// format is deliberately not wired here") for why `PTQ1_0`'s
    /// base-3-trit block has no compatible INT8 kernel.
    fn gemv_ptq1_0(
        &self,
        blocks: &[BlockPTQ1_0],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        let kernel = gemv_ptq1_0_kernel(self.prism_tier());
        prism_gemv_par(blocks, input, output, n_rows, k, QK_PTQ1_0, kernel)
    }

    /// **Not** wired to [`Int8Tier`]: see [`Self::gemv_ptq1_0`].
    fn gemm_ptq1_0(
        &self,
        blocks: &[BlockPTQ1_0],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        let tier = self.prism_tier();
        prism_gemm_dispatch(
            blocks,
            input,
            output,
            m,
            n_rows,
            k,
            QK_PTQ1_0,
            gemv_ptq1_0_kernel(tier),
            gemm_ptq1_0_kernel(tier),
        )
    }

    fn gemv_q2_0_g64(
        &self,
        blocks: &[BlockQ2_0G64],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        // See `gemv_pq2_0` above.
        if let Some(tier) = Int8Tier::from_env() {
            return dispatch_int8::gemv_two_bit_int8(tier, blocks, input, output, n_rows, k);
        }
        let kernel = gemv_q2_0_g64_kernel(self.prism_tier());
        prism_gemv_par(blocks, input, output, n_rows, k, QK_Q2_0_G64, kernel)
    }

    fn gemm_q2_0_g64(
        &self,
        blocks: &[BlockQ2_0G64],
        input: &[f32],
        output: &mut [f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        // See `gemv_pq2_0` above.
        if let Some(tier) = Int8Tier::from_env() {
            return dispatch_int8::gemm_two_bit_int8(tier, blocks, input, output, m, n_rows, k);
        }
        let tier = self.prism_tier();
        prism_gemm_dispatch(
            blocks,
            input,
            output,
            m,
            n_rows,
            k,
            QK_Q2_0_G64,
            gemv_q2_0_g64_kernel(tier),
            gemm_q2_0_g64_kernel(tier),
        )
    }

    // ── Hybrid math primitives: every tier converges on one self-dispatching
    // free function (see this module's doc comment) — no per-tier match.

    fn fwht_forward_signed(&self, x: &mut [f32], signs: &[f32], block: usize) -> KernelResult<()> {
        crate::hadamard::fwht_forward_signed(x, signs, block)
    }

    fn fwht_inverse_signed(&self, x: &mut [f32], signs: &[f32], block: usize) -> KernelResult<()> {
        crate::hadamard::fwht_inverse_signed(x, signs, block)
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_step(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha_raw: &[f32],
        beta_raw: &[f32],
        a_neg: &[f32],
        dt_bias: &[f32],
        state: &mut GdnState,
        layer: usize,
        out: &mut [f32],
    ) -> KernelResult<()> {
        crate::gated_delta_net::gdn_step(
            q, k, v, alpha_raw, beta_raw, a_neg, dt_bias, state, layer, out,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_chunk(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha_raw: &[f32],
        beta_raw: &[f32],
        a_neg: &[f32],
        dt_bias: &[f32],
        state: &mut GdnState,
        layer: usize,
        out: &mut [f32],
        t_len: usize,
    ) -> KernelResult<()> {
        crate::gated_delta_net_chunk::gdn_chunk(
            q, k, v, alpha_raw, beta_raw, a_neg, dt_bias, state, layer, out, t_len,
        )
    }

    fn conv1d_decode(
        &self,
        state: &mut [f32],
        x_t: &[f32],
        w: &[f32],
        out: &mut [f32],
    ) -> KernelResult<()> {
        crate::ssm_ops::causal_conv1d_k4_decode(state, x_t, w, out)
    }

    fn conv1d_prefill(
        &self,
        conv_x: &[f32],
        w: &[f32],
        out: &mut [f32],
        n_t: usize,
        d_inner: usize,
    ) -> KernelResult<()> {
        crate::ssm_ops::causal_conv1d_k4_prefill(conv_x, w, out, n_t, d_inner)
    }

    fn l2_norm(&self, input: &[f32], output: &mut [f32], eps: f32) -> KernelResult<()> {
        crate::norms::l2_norm_simd(input, output, eps)
    }

    fn rms_norm_gated(
        &self,
        input: &[f32],
        weight: &[f32],
        gate: &[f32],
        output: &mut [f32],
        eps: f32,
    ) -> KernelResult<()> {
        crate::norms::rms_norm_gated_simd(input, weight, gate, output, eps)
    }

    fn sigmoid_mul(&self, x: &[f32], gate: &[f32], output: &mut [f32]) -> KernelResult<()> {
        crate::norms::sigmoid_mul_simd(x, gate, output)
    }

    fn softplus(&self, input: &[f32], output: &mut [f32]) -> KernelResult<()> {
        crate::norms::softplus_simd(input, output)
    }

    fn rope_partial_splithalf(
        &self,
        input: &[f32],
        output: &mut [f32],
        head_dim: usize,
        n_rot: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> KernelResult<()> {
        crate::rope_mrope::rope_partial_splithalf_simd(input, output, head_dim, n_rot, cos, sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// Guards every test in this module that calls a wired `PrismKernel`
    /// method (`gemv_pq2_0`/`gemm_pq2_0`/`gemv_q2_0_g64`/`gemm_q2_0_g64`),
    /// which now reads [`crate::dispatch_int8::Int8Tier::from_env`] on
    /// every call. Returns [`crate::dispatch_int8::TierEnvGuard`], the same
    /// lock `dispatch_int8.rs`'s env-mutating test takes, which additionally
    /// snapshots/restores the variable so a panic here can't leak a
    /// mutated `OXIBONSAI_KERNEL_TIER` to later tests.
    fn env_guard() -> crate::dispatch_int8::TierEnvGuard {
        crate::dispatch_int8::TierEnvGuard::acquire()
    }

    fn pq2_0_block(scale: f32) -> BlockPQ2_0 {
        // All-zero `qs` decodes to code 0 -> value -1 everywhere (arithmetic
        // map `code - 1`), so a 128-wide all-ones input dots to `-128 * scale`.
        BlockPQ2_0 {
            d: f16::from_f32(scale),
            qs: [0u8; 32],
        }
    }

    #[test]
    fn gemv_pq2_0_reference_tier_matches_dequant_dot() {
        let _guard = env_guard();
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        let blocks = vec![pq2_0_block(1.0)];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];
        dispatcher
            .gemv_pq2_0(&blocks, &input, &mut output, 1, 128)
            .expect("gemv_pq2_0 should succeed");
        assert!(
            (output[0] + 128.0).abs() < 1e-3,
            "expected -128.0, got {}",
            output[0]
        );
    }

    #[test]
    fn gemm_pq2_0_batches_gemv_pq2_0() {
        let _guard = env_guard();
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);
        let blocks = vec![pq2_0_block(1.0)];
        let input = vec![1.0f32; 256]; // 2 rows of 128
        let mut output = vec![0.0f32; 2];
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut output, 2, 1, 128)
            .expect("gemm_pq2_0 should succeed");
        assert!((output[0] + 128.0).abs() < 1e-3);
        assert!((output[1] + 128.0).abs() < 1e-3);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_gemv_pq2_0_routes_through_a_cpu_tier() {
        let _guard = env_guard();
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let blocks = vec![pq2_0_block(1.0)];
        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 1];
        dispatcher
            .gemv_pq2_0(&blocks, &input, &mut output, 1, 128)
            .expect("Gpu-tier gemv_pq2_0 must still succeed via the CPU fallback");
        assert!((output[0] + 128.0).abs() < 1e-3);
    }

    #[test]
    fn fwht_forward_signed_involution_round_trips() {
        // Forward then inverse must return the original vector (design
        // §2.2's involution property), on any tier — exercised here on
        // whatever tier this test process auto-detects.
        let dispatcher = KernelDispatcher::auto_detect();
        let original = vec![1.0f32, 2.0, -3.0, 4.5, -0.5, 6.0, 7.0, -8.0];
        let signs = vec![1.0f32, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];
        let mut x = original.clone();
        dispatcher
            .fwht_forward_signed(&mut x, &signs, 8)
            .expect("forward should succeed");
        dispatcher
            .fwht_inverse_signed(&mut x, &signs, 8)
            .expect("inverse should succeed");
        for (a, b) in original.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-4, "expected {a}, got {b}");
        }
    }

    #[test]
    fn softplus_matches_cutoff_contract() {
        let dispatcher = KernelDispatcher::auto_detect();
        let input = vec![25.0f32, 0.0];
        let mut output = vec![0.0f32; 2];
        dispatcher
            .softplus(&input, &mut output)
            .expect("softplus should succeed");
        // Above the 20.0 cutoff, softplus(x) == x exactly (the acceptance
        // criterion).
        assert!((output[0] - 25.0).abs() < 1e-6);
        // softplus(0) == ln(2).
        assert!((output[1] - std::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_mul_matches_manual_computation() {
        let dispatcher = KernelDispatcher::auto_detect();
        let x = vec![2.0f32, -1.0];
        let gate = vec![0.0f32, 0.0]; // sigmoid(0) = 0.5
        let mut out = vec![0.0f32; 2];
        dispatcher
            .sigmoid_mul(&x, &gate, &mut out)
            .expect("sigmoid_mul should succeed");
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] + 0.5).abs() < 1e-5);
    }
}

/// Bit-exactness and routing guards for the register-blocked + Rayon Prism
/// GEMM/GEMV path.
///
/// Every comparison here is `assert_eq!` on raw `f32` values, never a
/// tolerance: register blocking and a Rayon split are both defined to leave
/// each output element's arithmetic untouched, so any drift is a bug, not
/// rounding. Module name contains `prism` so the package gate's `prism`
/// test-name filter selects it.
#[cfg(test)]
mod prism_blocked_tests {
    use super::*;
    use half::f16;

    /// See `tests::env_guard` (this file's other `#[cfg(test)]` module) —
    /// same guard, same reason: any test here that goes through the
    /// dispatcher (`KernelDispatcher::gemv_pq2_0` etc., not the raw
    /// `gemv_pq2_0_kernel(tier)(..)` functions) now reads
    /// `Int8Tier::from_env` and must be serialized against
    /// `dispatch_int8.rs`'s env-mutating test, with the environment
    /// snapshotted and restored around the call.
    fn env_guard() -> crate::dispatch_int8::TierEnvGuard {
        crate::dispatch_int8::TierEnvGuard::acquire()
    }

    /// Deterministic LCG — no `rand` dependency, reproducible across runs.
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
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 8) as i32 % 2001 - 1000) as f32 / 512.0
        }
    }

    fn pq2_blocks(n: usize, seed: u32) -> Vec<BlockPQ2_0> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let mut qs = [0u8; 32];
                for b in qs.iter_mut() {
                    *b = rng.next_u8();
                }
                BlockPQ2_0 {
                    d: f16::from_f32(0.125 + (rng.next_u8() % 8) as f32 / 64.0),
                    qs,
                }
            })
            .collect()
    }

    fn q2_g64_blocks(n: usize, seed: u32) -> Vec<BlockQ2_0G64> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let mut qs = [0u8; 16];
                for b in qs.iter_mut() {
                    *b = rng.next_u8();
                }
                BlockQ2_0G64 {
                    d: f16::from_f32(0.0625 + (rng.next_u8() % 8) as f32 / 64.0),
                    qs,
                }
            })
            .collect()
    }

    fn ptq1_blocks(n: usize, seed: u32) -> Vec<BlockPTQ1_0> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let mut qs = [0u8; 24];
                for b in qs.iter_mut() {
                    *b = rng.next_u8();
                }
                let qh = [rng.next_u8(), rng.next_u8()];
                BlockPTQ1_0 {
                    qs,
                    qh,
                    d: f16::from_f32(0.25 + (rng.next_u8() % 8) as f32 / 64.0),
                }
            })
            .collect()
    }

    fn inputs(len: usize, seed: u32) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..len).map(|_| rng.next_f32()).collect()
    }

    /// Every Prism tier this build can actually execute.
    fn tiers() -> Vec<PrismTier> {
        let mut v = vec![PrismTier::Scalar];
        #[cfg(target_arch = "aarch64")]
        v.push(PrismTier::Neon);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            v.push(PrismTier::Avx2);
        }
        v
    }

    /// The GEMV sweep the blocked GEMM must reproduce bit for bit: one
    /// `gemv` call per batch row, exactly as every Prism GEMM did before
    /// the register-blocked path.
    fn gemv_sweep<B>(
        kernel: PrismGemvFn<B>,
        blocks: &[B],
        input: &[f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n_rows];
        for mi in 0..m {
            kernel(
                blocks,
                &input[mi * k..(mi + 1) * k],
                &mut out[mi * n_rows..(mi + 1) * n_rows],
                n_rows,
                k,
            )
            .expect("gemv sweep should succeed");
        }
        out
    }

    fn assert_bit_identical(expect: &[f32], got: &[f32], what: &str) {
        assert_eq!(expect.len(), got.len(), "{what}: length");
        for (i, (e, g)) in expect.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                e.to_bits(),
                g.to_bits(),
                "{what}: element {i} differs ({e} vs {g})"
            );
        }
    }

    /// `m = 13` exercises every register-block tail shape (8 + 4 + 1).
    const TAIL_M: usize = 13;

    #[test]
    fn pq2_0_blocked_gemm_is_bit_identical_to_the_gemv_sweep() {
        let (n_rows, k) = (5usize, 2 * QK_PQ2_0);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x5EED_0001);
        let input = inputs(TAIL_M * k, 0x1234_0001);
        for tier in tiers() {
            let expect = gemv_sweep(gemv_pq2_0_kernel(tier), &blocks, &input, TAIL_M, n_rows, k);
            let mut got = vec![0.0f32; TAIL_M * n_rows];
            gemm_pq2_0_kernel(tier)(&blocks, &input, &mut got, TAIL_M, n_rows, k)
                .expect("blocked gemm should succeed");
            assert_bit_identical(&expect, &got, &format!("pq2_0 blocked {tier:?}"));
        }
    }

    #[test]
    fn q2_0_g64_blocked_gemm_is_bit_identical_to_the_gemv_sweep() {
        let (n_rows, k) = (7usize, 3 * QK_Q2_0_G64);
        let blocks = q2_g64_blocks(n_rows * (k / QK_Q2_0_G64), 0x5EED_0002);
        let input = inputs(TAIL_M * k, 0x1234_0002);
        for tier in tiers() {
            let expect = gemv_sweep(
                gemv_q2_0_g64_kernel(tier),
                &blocks,
                &input,
                TAIL_M,
                n_rows,
                k,
            );
            let mut got = vec![0.0f32; TAIL_M * n_rows];
            gemm_q2_0_g64_kernel(tier)(&blocks, &input, &mut got, TAIL_M, n_rows, k)
                .expect("blocked gemm should succeed");
            assert_bit_identical(&expect, &got, &format!("q2_0_g64 blocked {tier:?}"));
        }
    }

    #[test]
    fn ptq1_0_blocked_gemm_is_bit_identical_to_the_gemv_sweep() {
        let (n_rows, k) = (6usize, 2 * QK_PTQ1_0);
        let blocks = ptq1_blocks(n_rows * (k / QK_PTQ1_0), 0x5EED_0003);
        let input = inputs(TAIL_M * k, 0x1234_0003);
        for tier in tiers() {
            let expect = gemv_sweep(gemv_ptq1_0_kernel(tier), &blocks, &input, TAIL_M, n_rows, k);
            let mut got = vec![0.0f32; TAIL_M * n_rows];
            gemm_ptq1_0_kernel(tier)(&blocks, &input, &mut got, TAIL_M, n_rows, k)
                .expect("blocked gemm should succeed");
            assert_bit_identical(&expect, &got, &format!("ptq1_0 blocked {tier:?}"));
        }
    }

    /// The Rayon batch split must not change a single bit: `m` is chosen
    /// above `par_gemm_min_batch` on every platform profile this crate
    /// computes (its documented ceiling is 16).
    #[test]
    fn prism_gemm_par_matches_the_sequential_blocked_gemm_bit_for_bit() {
        let (m, n_rows, k) = (64usize, 9usize, 2 * QK_PQ2_0);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x5EED_0004);
        let input = inputs(m * k, 0x1234_0004);
        for tier in tiers() {
            let blocked = gemm_pq2_0_kernel(tier);
            let mut expect = vec![0.0f32; m * n_rows];
            blocked(&blocks, &input, &mut expect, m, n_rows, k).expect("sequential blocked");
            let mut got = vec![0.0f32; m * n_rows];
            prism_gemm_par(&blocks, &input, &mut got, m, n_rows, k, QK_PQ2_0, blocked)
                .expect("parallel blocked");
            assert_bit_identical(&expect, &got, &format!("pq2_0 gemm_par {tier:?}"));
        }
    }

    /// The Rayon row split must not change a single bit either. `n_rows` is
    /// above `par_gemv_min_rows`'s documented ceiling (1024).
    #[test]
    fn prism_gemv_par_matches_the_sequential_gemv_bit_for_bit() {
        let (n_rows, k) = (1100usize, QK_PTQ1_0);
        let blocks = ptq1_blocks(n_rows * (k / QK_PTQ1_0), 0x5EED_0005);
        let input = inputs(k, 0x1234_0005);
        for tier in tiers() {
            let kernel = gemv_ptq1_0_kernel(tier);
            let mut expect = vec![0.0f32; n_rows];
            kernel(&blocks, &input, &mut expect, n_rows, k).expect("sequential gemv");
            let mut got = vec![0.0f32; n_rows];
            prism_gemv_par(&blocks, &input, &mut got, n_rows, k, QK_PTQ1_0, kernel)
                .expect("parallel gemv");
            assert_bit_identical(&expect, &got, &format!("ptq1_0 gemv_par {tier:?}"));
        }
    }

    /// The dispatcher entry point (the signature `oxibonsai-model` and
    /// the prefill driver call) must agree with the raw blocked kernel.
    #[test]
    fn prism_dispatcher_gemm_matches_the_blocked_kernel_bit_for_bit() {
        let _guard = env_guard();
        let dispatcher = KernelDispatcher::auto_detect();
        let tier = dispatcher.prism_tier();
        let (m, n_rows, k) = (64usize, 9usize, 2 * QK_PQ2_0);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x5EED_0006);
        let input = inputs(m * k, 0x1234_0006);
        let mut expect = vec![0.0f32; m * n_rows];
        gemm_pq2_0_kernel(tier)(&blocks, &input, &mut expect, m, n_rows, k).expect("blocked");
        let mut got = vec![0.0f32; m * n_rows];
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut got, m, n_rows, k)
            .expect("dispatcher gemm");
        assert_bit_identical(&expect, &got, "dispatcher gemm_pq2_0");
    }

    /// Validation must stay in front of the parallel split: a short `blocks`
    /// slice has to produce the same named error the sequential kernel
    /// produced, not a slicing panic inside a Rayon task.
    #[test]
    fn prism_gemv_par_reports_the_same_named_error_as_the_kernel() {
        let (n_rows, k) = (1100usize, QK_PQ2_0);
        let blocks = pq2_blocks(n_rows - 1, 0x5EED_0007); // one row short
        let input = inputs(k, 0x1234_0007);
        let mut out = vec![0.0f32; n_rows];
        let err = prism_gemv_par(
            &blocks,
            &input,
            &mut out,
            n_rows,
            k,
            QK_PQ2_0,
            gemv_pq2_0_kernel(PrismTier::Scalar),
        )
        .expect_err("must reject a short block slice");
        assert_eq!(err.buffer_name(), Some("blocks"));
    }

    /// Tiling must not have changed what a `PQ2_0` GEMM *means*: the
    /// arithmetic map is still `code - 1` (`00 -> -1`), so an all-zero `qs`
    /// against an all-ones input is `-128 * d` per block.
    #[test]
    fn prism_blocked_gemm_still_uses_the_arithmetic_code_map() {
        let _guard = env_guard();
        let blocks = vec![BlockPQ2_0 {
            d: f16::from_f32(1.0),
            qs: [0u8; 32],
        }];
        let input = vec![1.0f32; 2 * QK_PQ2_0];
        let mut out = vec![0.0f32; 2];
        KernelDispatcher::with_tier(KernelTier::Reference)
            .gemm_pq2_0(&blocks, &input, &mut out, 2, 1, QK_PQ2_0)
            .expect("gemm_pq2_0");
        assert!((out[0] + 128.0).abs() < 1e-3, "got {}", out[0]);
        assert!((out[1] + 128.0).abs() < 1e-3, "got {}", out[1]);
    }

    /// Regression coverage: `m == 1` (decode) must route through
    /// weight-row-parallel `gemv_pq2_0` (via [`prism_gemm_dispatch`]), not
    /// the single-threaded blocked GEMM path — and must be bit-for-bit
    /// identical either way. `n_rows` is sized off `par_gemv_min_rows`
    /// (never hardcoded) so `prism_gemv_par`'s own Rayon split is really
    /// exercised, not just its below-threshold direct path.
    #[test]
    fn gemm_pq2_0_dispatcher_m1_matches_gemv_par() {
        let _guard = env_guard();
        let dispatcher = KernelDispatcher::auto_detect();
        let n_rows = PlatformProfile::global_thresholds().par_gemv_min_rows * 3 + 7;
        let k = 2 * QK_PQ2_0;
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x5EED_0010);
        let input = inputs(k, 0x1234_0010);

        let mut via_gemv = vec![0.0f32; n_rows];
        dispatcher
            .gemv_pq2_0(&blocks, &input, &mut via_gemv, n_rows, k)
            .expect("gemv_pq2_0");

        let mut via_gemm = vec![0.0f32; n_rows];
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut via_gemm, 1, n_rows, k)
            .expect("gemm_pq2_0 with m=1");

        assert_bit_identical(&via_gemv, &via_gemm, "gemm_pq2_0(m=1) vs gemv_pq2_0");
    }

    /// Same contract at `m == 5`, inside `(1, PRISM_GEMM_MR)` — the range
    /// that used to collapse to a single slab regardless of thread count
    /// (see [`prism_gemm_dispatch`]'s doc comment).
    #[test]
    fn gemm_pq2_0_dispatcher_m5_matches_gemv_sweep() {
        let _guard = env_guard();
        let dispatcher = KernelDispatcher::auto_detect();
        let n_rows = PlatformProfile::global_thresholds().par_gemv_min_rows * 2 + 3;
        let k = 2 * QK_PQ2_0;
        let m = 5usize;
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x5EED_0011);
        let input = inputs(m * k, 0x1234_0011);

        let mut expect = vec![0.0f32; m * n_rows];
        for mi in 0..m {
            dispatcher
                .gemv_pq2_0(
                    &blocks,
                    &input[mi * k..(mi + 1) * k],
                    &mut expect[mi * n_rows..(mi + 1) * n_rows],
                    n_rows,
                    k,
                )
                .expect("gemv sweep row");
        }

        let mut got = vec![0.0f32; m * n_rows];
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut got, m, n_rows, k)
            .expect("gemm_pq2_0 with m=5");

        assert_bit_identical(&expect, &got, "gemm_pq2_0(m=5) vs the gemv sweep");
    }

    /// Boundary case for [`prism_gemm_dispatch`]: `m == PRISM_GEMM_MR`
    /// exactly (not `m < PRISM_GEMM_MR`) is the case that was still missed
    /// by an earlier, off-by-one version of this dispatcher — at `m == MR`,
    /// `prism_gemm_chunk_rows`'s own floor makes `chunk_rows == m` too, so
    /// the *old* boundary (`m < MR`) would still hand this exact `m` to the
    /// single-slab blocked path. Bit-exactness is the only thing a unit
    /// test can pin cheaply; the parallelism win itself is measured by the
    /// `#[ignore]`d wall-clock tests.
    #[test]
    fn gemm_pq2_0_dispatcher_at_m_equals_mr_matches_gemv_sweep() {
        let _guard = env_guard();
        let dispatcher = KernelDispatcher::auto_detect();
        let n_rows = PlatformProfile::global_thresholds().par_gemv_min_rows * 2 + 5;
        let k = 2 * QK_PQ2_0;
        let m = PRISM_GEMM_MR; // exactly the boundary this dispatch logic must cover
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x5EED_0012);
        let input = inputs(m * k, 0x1234_0012);

        let mut expect = vec![0.0f32; m * n_rows];
        for mi in 0..m {
            dispatcher
                .gemv_pq2_0(
                    &blocks,
                    &input[mi * k..(mi + 1) * k],
                    &mut expect[mi * n_rows..(mi + 1) * n_rows],
                    n_rows,
                    k,
                )
                .expect("gemv sweep row");
        }

        let mut got = vec![0.0f32; m * n_rows];
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut got, m, n_rows, k)
            .expect("gemm_pq2_0 with m=PRISM_GEMM_MR");

        assert_bit_identical(&expect, &got, "gemm_pq2_0(m=MR) vs the gemv sweep");
    }

    /// The three tests just above compare the
    /// dispatcher against a manual GEMV *sweep* (one `gemv` call per batch
    /// row) — that is exactly what `prism_gemm_dispatch`
    /// itself does for `m <= PRISM_GEMM_MR`, so those tests exercise the
    /// dispatcher against its own equivalent, not against the raw
    /// register-blocked kernel (`gemm_pq2_0_kernel(tier)` etc.) it defers to
    /// for `m > PRISM_GEMM_MR`. This is the missing direct comparison: for
    /// every tier this build can execute and every `m` in `1..=PRISM_GEMM_MR`
    /// — the whole range the register-blocked fast path covers — the dispatcher's
    /// `gemm_{pq2_0,ptq1_0,q2_0_g64}` must still agree, bit for bit, with
    /// calling the *blocked* kernel function directly at that same `m` (the
    /// `*_blocked_gemm_is_bit_identical_to_the_gemv_sweep` tests already pin
    /// that the blocked kernel and the sweep agree at every `m`, so
    /// transitively so must the dispatcher and the blocked kernel — this
    /// test proves it is not merely transitively true but actually true of
    /// the real dispatch code path).
    ///
    /// `KernelDispatcher::with_tier` pins a tier, but a host that cannot
    /// execute it (e.g. `Avx2` off an `x86_64` box without the ISA) would
    /// silently clamp to a different tier — comparing scalar against Neon
    /// bits would then fail for the wrong reason, so this asserts
    /// `dispatcher.prism_tier() == tier` before trusting the comparison.
    #[test]
    fn dispatcher_gemm_agrees_with_the_blocked_kernel_for_every_m_up_to_mr() {
        let _guard = env_guard();
        let n_rows = PlatformProfile::global_thresholds().par_gemv_min_rows * 2 + 3;

        for tier in tiers() {
            let kernel_tier = match tier {
                PrismTier::Scalar => KernelTier::Reference,
                #[cfg(target_arch = "aarch64")]
                PrismTier::Neon => KernelTier::Neon,
                #[cfg(target_arch = "x86_64")]
                PrismTier::Avx2 => KernelTier::Avx2,
            };
            let dispatcher = KernelDispatcher::with_tier(kernel_tier);
            assert_eq!(
                dispatcher.prism_tier(),
                tier,
                "host cannot actually execute {tier:?} — this comparison would silently \
                 test the wrong tier"
            );

            for m in 1..=PRISM_GEMM_MR {
                // PQ2_0
                let k = 2 * QK_PQ2_0;
                let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0xA5A5_0000 ^ m as u32);
                let input = inputs(m * k, 0x5A5A_0000 ^ m as u32);
                let mut via_dispatcher = vec![0.0f32; m * n_rows];
                dispatcher
                    .gemm_pq2_0(&blocks, &input, &mut via_dispatcher, m, n_rows, k)
                    .expect("dispatcher gemm_pq2_0");
                let mut via_kernel = vec![0.0f32; m * n_rows];
                gemm_pq2_0_kernel(tier)(&blocks, &input, &mut via_kernel, m, n_rows, k)
                    .expect("blocked gemm_pq2_0_kernel");
                assert_bit_identical(
                    &via_dispatcher,
                    &via_kernel,
                    &format!("pq2_0 tier={tier:?} m={m}"),
                );

                // PTQ1_0
                let k_ptq1 = 2 * QK_PTQ1_0;
                let ptq1 = ptq1_blocks(n_rows * (k_ptq1 / QK_PTQ1_0), 0xB5B5_0000 ^ m as u32);
                let input_ptq1 = inputs(m * k_ptq1, 0x5B5B_0000 ^ m as u32);
                let mut via_dispatcher_ptq1 = vec![0.0f32; m * n_rows];
                dispatcher
                    .gemm_ptq1_0(
                        &ptq1,
                        &input_ptq1,
                        &mut via_dispatcher_ptq1,
                        m,
                        n_rows,
                        k_ptq1,
                    )
                    .expect("dispatcher gemm_ptq1_0");
                let mut via_kernel_ptq1 = vec![0.0f32; m * n_rows];
                gemm_ptq1_0_kernel(tier)(
                    &ptq1,
                    &input_ptq1,
                    &mut via_kernel_ptq1,
                    m,
                    n_rows,
                    k_ptq1,
                )
                .expect("blocked gemm_ptq1_0_kernel");
                assert_bit_identical(
                    &via_dispatcher_ptq1,
                    &via_kernel_ptq1,
                    &format!("ptq1_0 tier={tier:?} m={m}"),
                );

                // Q2_0G64
                let k_g64 = 2 * QK_Q2_0_G64;
                let g64 = q2_g64_blocks(n_rows * (k_g64 / QK_Q2_0_G64), 0xC5C5_0000 ^ m as u32);
                let input_g64 = inputs(m * k_g64, 0x5C5C_0000 ^ m as u32);
                let mut via_dispatcher_g64 = vec![0.0f32; m * n_rows];
                dispatcher
                    .gemm_q2_0_g64(&g64, &input_g64, &mut via_dispatcher_g64, m, n_rows, k_g64)
                    .expect("dispatcher gemm_q2_0_g64");
                let mut via_kernel_g64 = vec![0.0f32; m * n_rows];
                gemm_q2_0_g64_kernel(tier)(&g64, &input_g64, &mut via_kernel_g64, m, n_rows, k_g64)
                    .expect("blocked gemm_q2_0_g64_kernel");
                assert_bit_identical(
                    &via_dispatcher_g64,
                    &via_kernel_g64,
                    &format!("q2_0_g64 tier={tier:?} m={m}"),
                );
            }
        }
    }

    /// Numeric acceptance: a batched CPU prefill of
    /// Bonsai 2 27B shape must stop being a single-threaded GEMV loop.
    ///
    /// Shape: `ffn_up [5120, 17408]` (the 27B's widest per-layer matrix) at
    /// `M = 64`, one plausible prefill chunk — a **synthetic** matrix of that
    /// shape, not the 7.2 GB file, so the measurement is reproducible without
    /// model weights. `before` is the exact loop-of-GEMVs every Prism GEMM
    /// was before the register-blocked path; `after` is the dispatcher entry point the prefill driver calls.
    ///
    /// `#[ignore]` for the same reason as
    /// `gemv_ptq1::prism_gemv_tests::prism_ptq1_0_gemv_within_15pct_of_tq2`:
    /// it is a wall-clock comparison, meaningless in a debug build and
    /// unreliable while other work shares the machine. Run it deliberately:
    /// `cargo test -p oxibonsai-kernels --release --all-features
    /// prism_prefill_m64 -- --ignored --nocapture`.
    #[test]
    #[ignore = "wall-clock perf comparison; run manually in --release, not under concurrent-build CI"]
    fn prism_prefill_m64_is_no_longer_a_single_threaded_gemv_loop() {
        use std::time::Instant;

        let _guard = env_guard();
        let (m, n_rows, k) = (64usize, 17408usize, 5120usize);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0xB00C_0001);
        let input = inputs(m * k, 0xB00C_0002);
        let dispatcher = KernelDispatcher::auto_detect();
        let tier = dispatcher.prism_tier();
        let sweep = gemv_pq2_0_kernel(tier);

        // Warm the caches/branch predictors once on each path.
        let mut before_out = vec![0.0f32; m * n_rows];
        let mut after_out = vec![0.0f32; m * n_rows];
        for mi in 0..2 {
            sweep(
                &blocks,
                &input[mi * k..(mi + 1) * k],
                &mut before_out[mi * n_rows..(mi + 1) * n_rows],
                n_rows,
                k,
            )
            .expect("warmup sweep");
        }
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut after_out, m, n_rows, k)
            .expect("warmup gemm");

        let t0 = Instant::now();
        for mi in 0..m {
            sweep(
                &blocks,
                &input[mi * k..(mi + 1) * k],
                &mut before_out[mi * n_rows..(mi + 1) * n_rows],
                n_rows,
                k,
            )
            .expect("before: gemv sweep");
        }
        let before = t0.elapsed();

        let t1 = Instant::now();
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut after_out, m, n_rows, k)
            .expect("after: tiled + rayon gemm");
        let after = t1.elapsed();

        let speedup = before.as_secs_f64() / after.as_secs_f64().max(1e-9);
        println!(
            "prism prefill M=64 ffn_up[{k},{n_rows}] tier={tier:?} threads={}: \
             before {:?} -> after {:?} = {speedup:.2}x",
            rayon::current_num_threads(),
            before,
            after
        );
        assert!(
            speedup >= 4.0,
            "tiled + rayon Prism GEMM must be >= 4x the single-threaded GEMV loop \
             at M=64 (got {speedup:.2}x: {before:?} -> {after:?})"
        );
    }

    /// Numeric acceptance: the same before/after
    /// shape as [`prism_prefill_m64_is_no_longer_a_single_threaded_gemv_loop`],
    /// at `M = 1` (decode) instead of `64` — `before` is the plain blocked
    /// kernel (what every `gemm_pq2_0` call used to run at this `m`, since
    /// `1 < par_gemm_min_batch` on every platform profile), `after` is
    /// [`prism_gemm_dispatch`]'s `m <= PRISM_GEMM_MR` fast path.
    ///
    /// Takes the **min of several trials** on each side rather than one
    /// shot: this dev machine runs under heavy, highly variable background
    /// load from sibling sessions (observed load average 30-60 on 8 cores
    /// while developing this test), and a single sample swings from ~0.6x
    /// to ~4x depending on what else the scheduler is doing to the process
    /// in that instant — the min isolates the code path's own cost from
    /// that noise (standard microbenchmark practice under contention).
    /// Observed range across repeated runs while developing this fix:
    /// 1.4x-3.9x; the assertion below picks a floor comfortably under the
    /// worst of those, not the best case. Same `#[ignore]` reason as the
    /// `M = 64` test: a wall-clock comparison, meaningless in a debug
    /// build, and still sensitive to how loaded the machine is. Run
    /// deliberately: `cargo test -p oxibonsai-kernels --release
    /// --all-features prism_decode_m1 -- --ignored --nocapture`.
    #[test]
    #[ignore = "wall-clock perf comparison; run manually in --release, not under concurrent-build CI"]
    fn prism_decode_m1_is_no_longer_a_single_threaded_blocked_task() {
        use std::time::{Duration, Instant};

        /// Wall-clock min of `TRIALS` calls to `f`, discarding scheduler
        /// noise spikes rather than averaging them in.
        fn min_of_trials(trials: u32, mut f: impl FnMut()) -> Duration {
            let mut best = Duration::MAX;
            for _ in 0..trials {
                let t0 = Instant::now();
                f();
                best = best.min(t0.elapsed());
            }
            best
        }

        let _guard = env_guard();
        let (n_rows, k) = (17408usize, 5120usize);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0xB00C_0011);
        let input = inputs(k, 0xB00C_0012);
        let dispatcher = KernelDispatcher::auto_detect();
        let tier = dispatcher.prism_tier();
        let blocked = gemm_pq2_0_kernel(tier);
        const TRIALS: u32 = 15;

        // Warm the caches/branch predictors once on each path before timing.
        let mut before_out = vec![0.0f32; n_rows];
        let mut after_out = vec![0.0f32; n_rows];
        blocked(&blocks, &input, &mut before_out, 1, n_rows, k).expect("warmup blocked");
        dispatcher
            .gemm_pq2_0(&blocks, &input, &mut after_out, 1, n_rows, k)
            .expect("warmup dispatcher");

        let before = min_of_trials(TRIALS, || {
            blocked(&blocks, &input, &mut before_out, 1, n_rows, k)
                .expect("before: single-threaded blocked m=1");
        });
        let after = min_of_trials(TRIALS, || {
            dispatcher
                .gemm_pq2_0(&blocks, &input, &mut after_out, 1, n_rows, k)
                .expect("after: dispatcher gemm_pq2_0 m=1");
        });

        assert_bit_identical(
            &before_out,
            &after_out,
            "before/after m=1 must still agree bit for bit",
        );

        let speedup = before.as_secs_f64() / after.as_secs_f64().max(1e-9);
        println!(
            "prism decode M=1 ffn_up[{k},{n_rows}] tier={tier:?} threads={}: \
             before {:?} -> after {:?} = {speedup:.2}x (min of {TRIALS} trials each)",
            rayon::current_num_threads(),
            before,
            after
        );
        assert!(
            speedup >= 1.3,
            "gemm_pq2_0(m=1) must be faster once routed through prism_gemv_par \
             (got {speedup:.2}x: {before:?} -> {after:?}); if this fails on a \
             quiet machine (not just under shared-CI-style load), that is a \
             real regression, not noise"
        );
    }

    /// The three **GEMM** entry points must record
    /// the CPU tier a `KernelTier::Gpu` dispatcher fell back onto, which
    /// only the GEMV ones did before.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_prism_gemms_record_the_cpu_fallback_tier() {
        use crate::dispatch::LAST_GPU_FALLBACK_TIER;
        let _guard = env_guard();
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let expect = KernelDispatcher::cpu_tier();

        let pq2 = pq2_blocks(1, 0x5EED_0008);
        let input = inputs(2 * QK_PQ2_0, 0x1234_0008);
        let mut out = vec![0.0f32; 2];
        LAST_GPU_FALLBACK_TIER.with(|c| c.set(None));
        dispatcher
            .gemm_pq2_0(&pq2, &input, &mut out, 2, 1, QK_PQ2_0)
            .expect("gemm_pq2_0");
        assert_eq!(LAST_GPU_FALLBACK_TIER.with(|c| c.get()), Some(expect));

        let ptq1 = ptq1_blocks(1, 0x5EED_0009);
        let mut out = vec![0.0f32; 2];
        LAST_GPU_FALLBACK_TIER.with(|c| c.set(None));
        dispatcher
            .gemm_ptq1_0(&ptq1, &input, &mut out, 2, 1, QK_PTQ1_0)
            .expect("gemm_ptq1_0");
        assert_eq!(LAST_GPU_FALLBACK_TIER.with(|c| c.get()), Some(expect));

        let g64 = q2_g64_blocks(1, 0x5EED_000A);
        let input64 = inputs(2 * QK_Q2_0_G64, 0x1234_000A);
        let mut out = vec![0.0f32; 2];
        LAST_GPU_FALLBACK_TIER.with(|c| c.set(None));
        dispatcher
            .gemm_q2_0_g64(&g64, &input64, &mut out, 2, 1, QK_Q2_0_G64)
            .expect("gemm_q2_0_g64");
        assert_eq!(LAST_GPU_FALLBACK_TIER.with(|c| c.get()), Some(expect));
    }

    // ─── End-to-end INT8 tier wiring ───────────────────────────────────

    /// Acceptance: `OXIBONSAI_KERNEL_TIER` must
    /// change `gemv_pq2_0`/`gemm_pq2_0`'s actual output, by producing
    /// exactly what calling `dispatch_int8::{gemv,gemm}_two_bit_int8`
    /// directly would (the wiring is a straight delegation, not a second
    /// implementation to drift out of sync) — and that output must differ
    /// from the plain f32 path once the variable is cleared again, or this
    /// test would prove the dispatcher never actually reads its tier.
    #[test]
    fn gemv_and_gemm_pq2_0_route_through_the_int8_tier_when_asked() {
        let _guard = env_guard();
        let (n_rows, k) = (5usize, 2 * QK_PQ2_0);
        let blocks = pq2_blocks(n_rows * (k / QK_PQ2_0), 0x9EED_0001);
        let input = inputs(k, 0x1234_1001);
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);

        // SAFETY: serialized by `env_guard()` above.
        unsafe {
            std::env::set_var(dispatch_int8::KERNEL_TIER_ENV, "int8-scalar");
        }

        let mut env_selected = vec![0.0f32; n_rows];
        dispatcher
            .gemv_pq2_0(&blocks, &input, &mut env_selected, n_rows, k)
            .expect("env-selected gemv_pq2_0");
        let mut direct_int8 = vec![0.0f32; n_rows];
        dispatch_int8::gemv_two_bit_int8(
            Int8Tier::Scalar,
            &blocks,
            &input,
            &mut direct_int8,
            n_rows,
            k,
        )
        .expect("direct int8 gemv");
        assert_bit_identical(
            &direct_int8,
            &env_selected,
            "env-selected gemv_pq2_0 vs direct dispatch_int8::gemv_two_bit_int8",
        );

        let batched_input = inputs(3 * k, 0x1234_1002);
        let mut env_selected_gemm = vec![0.0f32; 3 * n_rows];
        dispatcher
            .gemm_pq2_0(
                &blocks,
                &batched_input,
                &mut env_selected_gemm,
                3,
                n_rows,
                k,
            )
            .expect("env-selected gemm_pq2_0");
        let mut direct_int8_gemm = vec![0.0f32; 3 * n_rows];
        dispatch_int8::gemm_two_bit_int8(
            Int8Tier::Scalar,
            &blocks,
            &batched_input,
            &mut direct_int8_gemm,
            3,
            n_rows,
            k,
        )
        .expect("direct int8 gemm");
        assert_bit_identical(
            &direct_int8_gemm,
            &env_selected_gemm,
            "env-selected gemm_pq2_0 vs direct dispatch_int8::gemm_two_bit_int8",
        );

        // SAFETY: serialized by `env_guard()` above.
        unsafe {
            std::env::remove_var(dispatch_int8::KERNEL_TIER_ENV);
        }

        let mut f32_path = vec![0.0f32; n_rows];
        dispatcher
            .gemv_pq2_0(&blocks, &input, &mut f32_path, n_rows, k)
            .expect("f32 gemv_pq2_0 after clearing the env var");
        assert!(
            f32_path
                .iter()
                .zip(env_selected.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "the int8-tier GEMV result was bit-identical to the f32 result -- \
             the env var did not actually change which kernel ran"
        );

        // The GEMV divergence check
        // above does not prove `gemm_pq2_0` reads the tier too -- only the
        // bit-identity-to-`dispatch_int8` checks earlier in this test did
        // that, and only against the int8 path. Prove GEMM also diverges
        // from the f32 path once the env var is cleared.
        let mut f32_path_gemm = vec![0.0f32; 3 * n_rows];
        dispatcher
            .gemm_pq2_0(&blocks, &batched_input, &mut f32_path_gemm, 3, n_rows, k)
            .expect("f32 gemm_pq2_0 after clearing the env var");
        assert!(
            f32_path_gemm
                .iter()
                .zip(env_selected_gemm.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "the int8-tier GEMM result was bit-identical to the f32 GEMM result -- \
             the env var did not actually change which kernel ran"
        );
    }

    /// See [`gemv_and_gemm_pq2_0_route_through_the_int8_tier_when_asked`];
    /// same contract for the mainline group-64 `Q2_0` format.
    #[test]
    fn gemv_and_gemm_q2_0_g64_route_through_the_int8_tier_when_asked() {
        let _guard = env_guard();
        let (n_rows, k) = (5usize, 3 * QK_Q2_0_G64);
        let blocks = q2_g64_blocks(n_rows * (k / QK_Q2_0_G64), 0x9EED_0002);
        let input = inputs(k, 0x1234_2001);
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);

        // SAFETY: serialized by `env_guard()` above.
        unsafe {
            std::env::set_var(dispatch_int8::KERNEL_TIER_ENV, "int8-scalar");
        }

        let mut env_selected = vec![0.0f32; n_rows];
        dispatcher
            .gemv_q2_0_g64(&blocks, &input, &mut env_selected, n_rows, k)
            .expect("env-selected gemv_q2_0_g64");
        let mut direct_int8 = vec![0.0f32; n_rows];
        dispatch_int8::gemv_two_bit_int8(
            Int8Tier::Scalar,
            &blocks,
            &input,
            &mut direct_int8,
            n_rows,
            k,
        )
        .expect("direct int8 gemv");
        assert_bit_identical(
            &direct_int8,
            &env_selected,
            "env-selected gemv_q2_0_g64 vs direct dispatch_int8::gemv_two_bit_int8",
        );

        let batched_input = inputs(3 * k, 0x1234_2002);
        let mut env_selected_gemm = vec![0.0f32; 3 * n_rows];
        dispatcher
            .gemm_q2_0_g64(
                &blocks,
                &batched_input,
                &mut env_selected_gemm,
                3,
                n_rows,
                k,
            )
            .expect("env-selected gemm_q2_0_g64");
        let mut direct_int8_gemm = vec![0.0f32; 3 * n_rows];
        dispatch_int8::gemm_two_bit_int8(
            Int8Tier::Scalar,
            &blocks,
            &batched_input,
            &mut direct_int8_gemm,
            3,
            n_rows,
            k,
        )
        .expect("direct int8 gemm");
        assert_bit_identical(
            &direct_int8_gemm,
            &env_selected_gemm,
            "env-selected gemm_q2_0_g64 vs direct dispatch_int8::gemm_two_bit_int8",
        );

        // SAFETY: serialized by `env_guard()` above.
        unsafe {
            std::env::remove_var(dispatch_int8::KERNEL_TIER_ENV);
        }

        // This test previously stopped
        // at the int8-vs-int8 bit-identity checks above and never proved
        // the env var changes anything relative to the plain f32 path (the
        // pq2_0 sibling test only checked this for GEMV). Check both GEMV
        // and GEMM diverge from the f32 path once the variable is cleared.
        let mut f32_path = vec![0.0f32; n_rows];
        dispatcher
            .gemv_q2_0_g64(&blocks, &input, &mut f32_path, n_rows, k)
            .expect("f32 gemv_q2_0_g64 after clearing the env var");
        assert!(
            f32_path
                .iter()
                .zip(env_selected.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "the int8-tier GEMV result was bit-identical to the f32 result -- \
             the env var did not actually change which kernel ran"
        );

        let mut f32_path_gemm = vec![0.0f32; 3 * n_rows];
        dispatcher
            .gemm_q2_0_g64(&blocks, &batched_input, &mut f32_path_gemm, 3, n_rows, k)
            .expect("f32 gemm_q2_0_g64 after clearing the env var");
        assert!(
            f32_path_gemm
                .iter()
                .zip(env_selected_gemm.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "the int8-tier GEMM result was bit-identical to the f32 GEMM result -- \
             the env var did not actually change which kernel ran"
        );
    }

    /// Pins the deliberate non-wiring this module's doc comment explains:
    /// `PTQ1_0` has no compatible `Int8TwoBitBlock` impl (its block is
    /// base-3-trit-packed, not 2-bit-code-packed), so
    /// `OXIBONSAI_KERNEL_TIER` must have **zero** effect on `gemv_ptq1_0`'s
    /// output, set or not.
    #[test]
    fn gemv_ptq1_0_ignores_the_int8_tier_env_var() {
        let _guard = env_guard();
        let (n_rows, k) = (6usize, 2 * QK_PTQ1_0);
        let blocks = ptq1_blocks(n_rows * (k / QK_PTQ1_0), 0x9EED_0003);
        let input = inputs(k, 0x1234_3001);
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Reference);

        let mut baseline = vec![0.0f32; n_rows];
        dispatcher
            .gemv_ptq1_0(&blocks, &input, &mut baseline, n_rows, k)
            .expect("gemv_ptq1_0 with a clean environment");

        // SAFETY: serialized by `env_guard()` above.
        unsafe {
            std::env::set_var(dispatch_int8::KERNEL_TIER_ENV, "int8-scalar");
        }
        let mut with_env_set = vec![0.0f32; n_rows];
        dispatcher
            .gemv_ptq1_0(&blocks, &input, &mut with_env_set, n_rows, k)
            .expect("gemv_ptq1_0 with the INT8 tier requested");
        // SAFETY: serialized by `env_guard()` above.
        unsafe {
            std::env::remove_var(dispatch_int8::KERNEL_TIER_ENV);
        }

        assert_bit_identical(
            &baseline,
            &with_env_set,
            "gemv_ptq1_0 must ignore OXIBONSAI_KERNEL_TIER entirely",
        );
    }
}
