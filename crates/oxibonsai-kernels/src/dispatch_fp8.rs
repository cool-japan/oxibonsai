//! [`Fp8Kernel`] dispatch: FP8 E4M3FN / E5M2 dequantization, GEMV and GEMM.
//!
//! Split out of `dispatch.rs` (1904 lines against its 2000-line budget) the
//! same way `dispatch_std_quant.rs` holds the `StandardQuantKernel` impl and
//! `dispatch_prism.rs` the `PrismKernel` one: a single `impl Trait for Type`
//! cannot itself be split across files, so every `Fp8Kernel` method — and
//! every FP8-only helper — lives here.
//!
//! ## The GPU GEMV kernels belong to `KernelTier::Gpu` alone
//!
//! `gemv_fp8_e4m3` / `gemv_fp8_e5m2` reach the Metal (`metal_gemv_fp8_*`,
//! macOS + `metal`) or CUDA (`cuda_gemv_fp8_*`, Linux/Windows +
//! `native-cuda`) kernel **only** from a `KernelTier::Gpu` dispatcher — the
//! gate every other quant family applies (`gemv_1bit_on_tier`,
//! `OneBitKernel::upload_weights`, `StandardQuantKernel::gemv_q4_0`, …), and
//! the behaviour the FP8 GPU GEMV was documented to have from the start
//! ("Metal → CUDA → CPU SIMD on the `Gpu` tier"). A CPU-tier dispatcher —
//! `Reference`, `Avx2`, `Avx512`, `Neon` — never enters the private `gpu`
//! module below at all.
//!
//! These two methods used to try Metal and CUDA *unconditionally*, ahead of
//! their tier `match`, so on an `--all-features` Linux build every CPU-tier
//! FP8 GEMV — `KernelDispatcher::with_tier(cpu_kernel_tier())`, the
//! `Reference` tier the FP8 kernel tests pin, the layers of an engine loaded
//! with `Backend::Cpu` (whose `CpuOnlyBackendScope` exists precisely to keep
//! every GEMV off the GPU) — went through `cuda_gemv_fp8_*`. On a host with
//! a GPU that meant a full weight upload, a kernel launch and a device-to-host
//! copy per call, with the result computed by the CUDA kernel instead of the
//! tier's own CPU kernel; on a host without one, a clone of the memoised
//! `CudaGraphError` plus an `e.to_string()` per call, just to conclude that
//! the error was benign. Either way the call allocated: the 2 allocations + 1
//! reallocation `tests/gemv_no_alloc.rs` flagged on both FP8 `[sequential]`
//! paths of its `Avx2` dispatcher.
//!
//! ## `KernelTier::Gpu` falls back to the CPU SIMD tier (K-17)
//!
//! A `Gpu` dispatcher's GEMV tries the GPU kernel this build carries first.
//! When the build carries none, when the GPU path declines the call (a shape
//! the kernels cannot or need not serve — see `gpu::weight_image` and
//! `gpu::gemv_operands`), or when the kernel fails, the call runs on
//! `KernelDispatcher::cpu_tier()` — AVX-512, AVX2 or NEON — and on the scalar
//! reference only when that genuinely is the best this CPU has.
//! Dequantization and GEMM have no per-call GPU kernel behind this dispatcher
//! (the Metal / CUDA FP8 batch kernels are driven by `oxibonsai-model`'s
//! whole-layer prefill paths, never through [`Fp8Kernel`]), so a `Gpu`
//! dispatcher routes them straight to that CPU SIMD tier too — the policy
//! `dispatch_prism.rs` documents for a format with no per-call GPU kernel.
//! (Their `Gpu` arm used to fall through `_ =>` to the scalar reference.)
//! Every such routing goes through one resolver,
//! `KernelDispatcher::fp8_cpu_tier`, which records it in `dispatch.rs`'s
//! `LAST_GPU_FALLBACK_TIER` under `cfg(test)`, exactly as
//! `dispatch_std_quant.rs` and `dispatch_prism.rs` record theirs.
//!
//! ## A missing device is recognised by type, not by message
//!
//! A GPU GEMV failing with `MetalGraphError::DeviceNotFound` or
//! `CudaGraphError::DeviceNotFound(_)` falls back silently — no device is the
//! expected state of a GPU-less host, not a fault — and every other error
//! keeps its `tracing::warn!`. The test used to be
//! `e.to_string().contains("no Metal-capable GPU device")` /
//! `.contains("no CUDA device")`: an allocation on every failed call, and a
//! silent dependency on two `Display` strings.

use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2};

use crate::dispatch::{KernelDispatcher, KernelTier};
use crate::error::KernelResult;
use crate::traits::Fp8Kernel;

// Compile-time layout checks: both FP8 block types must be exactly
// `BLOCK_FP8_BYTES` (34) bytes — `[q0..q31, scale_lo, scale_hi]`, the AoS
// stride both GPU GEMV kernels index — which the raw-byte view of a block
// slice in `gpu::gemv_operands` relies on.
const _: () = assert!(std::mem::size_of::<BlockFP8E4M3>() == oxibonsai_core::BLOCK_FP8_BYTES);
const _: () = assert!(std::mem::size_of::<BlockFP8E5M2>() == oxibonsai_core::BLOCK_FP8_BYTES);

/// Whether this build carries an FP8 GPU GEMV kernel for a `KernelTier::Gpu`
/// dispatcher to try: Metal on macOS with `metal`, or CUDA on Linux/Windows
/// with `native-cuda`. A build with `gpu` but neither of those (for example
/// `--features cuda`, the scirs2-core backend) has none, so its `Gpu`
/// dispatcher runs every FP8 call on the CPU SIMD tier.
///
/// That predicate is the `fp8_gpu_gemv` cfg, which `build.rs` sets for
/// exactly those builds and nowhere else: this constant is
/// `cfg!(fp8_gpu_gemv)`, and every gate in this module that needs the
/// predicate — the two GEMV call sites, the private `gpu` module, their
/// tests — is `#[cfg(fp8_gpu_gemv)]` rather than another copy of the
/// feature/OS combination. Code inside the GPU path that differs per backend
/// still names its own backend's predicate.
#[cfg(feature = "gpu")]
const FP8_GPU_GEMV_COMPILED: bool = cfg!(fp8_gpu_gemv);

// ─── Tier mapping ────────────────────────────────────────────────────────

/// Which per-tier FP8 CPU kernel family a [`KernelTier`] maps onto.
///
/// Deliberately a separate enum from [`KernelTier`], for the reason
/// `dispatch_prism.rs`'s `PrismTier` is: resolving `Gpu → best CPU tier`
/// once, in `KernelDispatcher::fp8_cpu_tier`, keeps every selector below a
/// plain match over the kernels that actually exist. Each SIMD variant only
/// exists on the architecture whose `#[target_feature]` kernels it selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fp8CpuTier {
    /// Scalar reference (`dequant_fp8` / `gemv_fp8` / `gemm_fp8`).
    Scalar,
    /// AVX2 + FMA (`simd_fp8_avx2`).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AVX-512 F + BW + VL (`simd_fp8_avx512`).
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// NEON (`simd_fp8_neon`).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Fp8CpuTier {
    /// The [`Fp8Kernel::name_fp8`] label of this kernel family.
    const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "fp8_reference",
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => "fp8_avx2",
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => "fp8_avx512",
            #[cfg(target_arch = "aarch64")]
            Self::Neon => "fp8_neon",
        }
    }
}

/// Map a [`KernelTier`] onto the FP8 CPU kernel family that executes it.
///
/// `KernelTier::Gpu` must be resolved by the caller
/// (`KernelDispatcher::fp8_cpu_tier`) before reaching here, because doing so
/// has an observable side effect (recording the K-17 fallback tier); it is
/// mapped to [`Fp8CpuTier::Scalar`] defensively so this function stays total,
/// as `dispatch_prism.rs`'s `prism_tier_of` does.
fn fp8_cpu_tier_of(tier: KernelTier) -> Fp8CpuTier {
    match tier {
        KernelTier::Reference => Fp8CpuTier::Scalar,
        #[cfg(target_arch = "x86_64")]
        KernelTier::Avx2 => Fp8CpuTier::Avx2,
        #[cfg(target_arch = "x86_64")]
        KernelTier::Avx512 => Fp8CpuTier::Avx512,
        #[cfg(target_arch = "aarch64")]
        KernelTier::Neon => Fp8CpuTier::Neon,
        #[cfg(feature = "gpu")]
        KernelTier::Gpu => Fp8CpuTier::Scalar,
    }
}

impl KernelDispatcher {
    /// The FP8 CPU kernel family this dispatcher's dequant and GEMM calls run
    /// on — and its GEMV calls, whenever a `KernelTier::Gpu` dispatcher's GPU
    /// attempt has not produced the result.
    ///
    /// A `Gpu` dispatcher resolves through `Self::cpu_tier()` (K-17: the best
    /// CPU SIMD tier, never a hardcoded scalar) and records that routing
    /// decision under `cfg(test)` so `dispatch.rs`'s `LAST_GPU_FALLBACK_TIER`
    /// sees it. This is the one place every FP8 entry point resolves its CPU
    /// tier, so none can drift back to scalar on its own. A CPU-tier
    /// dispatcher maps straight onto its own tier and records nothing.
    fn fp8_cpu_tier(&self) -> Fp8CpuTier {
        match self.tier() {
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                let cpu = Self::cpu_tier();
                #[cfg(test)]
                crate::dispatch::record_gpu_fallback_tier(cpu);
                fp8_cpu_tier_of(cpu)
            }
            tier => fp8_cpu_tier_of(tier),
        }
    }
}

// ─── Per-tier kernel selection ───────────────────────────────────────────

#[cfg(test)]
thread_local! {
    /// The FP8 CPU kernel family the most recent dequant / GEMV / GEMM on
    /// this thread actually ran — noted by the kernel a selector below
    /// returns, from the same match arm that names the kernel it calls, the
    /// moment it runs.
    ///
    /// The outputs alone cannot show which tier's kernel ran: dequantization
    /// is bit-identical on every tier (each element is one exact FP8 decode
    /// times the block scale), so a selector handing a SIMD tier the scalar
    /// dequant — or an entry point bypassing its selector altogether — would
    /// pass any bit-for-bit comparison. Thread-local, for the reason
    /// `dispatch.rs`'s `LAST_GPU_FALLBACK_TIER` is.
    static LAST_FP8_CPU_KERNEL: std::cell::Cell<Option<Fp8CpuTier>> =
        const { std::cell::Cell::new(None) };
}

/// Note that `tier`'s kernel is about to run (see `LAST_FP8_CPU_KERNEL`).
#[cfg(test)]
fn note_cpu_kernel(tier: Fp8CpuTier) {
    LAST_FP8_CPU_KERNEL.with(|last| last.set(Some(tier)));
}

/// Outside tests there is nothing to note: this compiles to nothing.
#[cfg(not(test))]
#[inline(always)]
fn note_cpu_kernel(_: Fp8CpuTier) {}

/// An FP8 dequantization kernel: `(blocks, output)`.
type Fp8DequantFn<B> = fn(&[B], &mut [f32]) -> KernelResult<()>;

/// An FP8 GEMV kernel: `(blocks, input, output, n_rows, k)`.
type Fp8GemvFn<B> = fn(&[B], &[f32], &mut [f32], usize, usize) -> KernelResult<()>;

/// An FP8 GEMM kernel: `(blocks, inputs, outputs, n_rows, k, batch)`.
type Fp8GemmFn<B> = fn(&[B], &[f32], &mut [f32], usize, usize, usize) -> KernelResult<()>;

/// SAFETY (shared by every `unsafe` block in the six selectors below): each
/// [`Fp8CpuTier`] SIMD variant only exists on the architecture whose
/// `#[target_feature]` kernels it selects, and is only ever produced from a
/// [`KernelTier`] the running CPU executes — either a dispatcher's own tier,
/// which every safe constructor re-validates (`clamp_tier_to_cpu`, with
/// `is_x86_feature_detected!`; `with_tier_unchecked`'s `# Safety` contract
/// makes its caller vouch for it instead), or `KernelDispatcher::cpu_tier()`,
/// which detects it at runtime. NEON is the AArch64 baseline.
///
/// Every arm of the six selectors returns a closure that first notes its own
/// kernel family (`note_cpu_kernel`, which compiles to nothing outside tests)
/// and then calls that family's kernel, so the tests can see which kernel a
/// dispatcher actually ran even where two tiers' outputs coincide bit for bit.
fn dequant_fp8_e4m3_kernel(tier: Fp8CpuTier) -> Fp8DequantFn<BlockFP8E4M3> {
    match tier {
        Fp8CpuTier::Scalar => |b, o| {
            note_cpu_kernel(Fp8CpuTier::Scalar);
            crate::dequant_fp8::dequant_fp8_e4m3(b, o)
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx2 => |b, o| {
            note_cpu_kernel(Fp8CpuTier::Avx2);
            unsafe { crate::simd_fp8_avx2::dequant_fp8_e4m3_avx2(b, o) }
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx512 => |b, o| {
            note_cpu_kernel(Fp8CpuTier::Avx512);
            unsafe { crate::simd_fp8_avx512::dequant_fp8_e4m3_avx512(b, o) }
        },
        #[cfg(target_arch = "aarch64")]
        Fp8CpuTier::Neon => |b, o| {
            note_cpu_kernel(Fp8CpuTier::Neon);
            unsafe { crate::simd_fp8_neon::dequant_fp8_e4m3_neon(b, o) }
        },
    }
}

/// See [`dequant_fp8_e4m3_kernel`] for the shared safety argument.
fn dequant_fp8_e5m2_kernel(tier: Fp8CpuTier) -> Fp8DequantFn<BlockFP8E5M2> {
    match tier {
        Fp8CpuTier::Scalar => |b, o| {
            note_cpu_kernel(Fp8CpuTier::Scalar);
            crate::dequant_fp8::dequant_fp8_e5m2(b, o)
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx2 => |b, o| {
            note_cpu_kernel(Fp8CpuTier::Avx2);
            unsafe { crate::simd_fp8_avx2::dequant_fp8_e5m2_avx2(b, o) }
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx512 => |b, o| {
            note_cpu_kernel(Fp8CpuTier::Avx512);
            unsafe { crate::simd_fp8_avx512::dequant_fp8_e5m2_avx512(b, o) }
        },
        #[cfg(target_arch = "aarch64")]
        Fp8CpuTier::Neon => |b, o| {
            note_cpu_kernel(Fp8CpuTier::Neon);
            unsafe { crate::simd_fp8_neon::dequant_fp8_e5m2_neon(b, o) }
        },
    }
}

/// See [`dequant_fp8_e4m3_kernel`] for the shared safety argument.
fn gemv_fp8_e4m3_kernel(tier: Fp8CpuTier) -> Fp8GemvFn<BlockFP8E4M3> {
    match tier {
        Fp8CpuTier::Scalar => |b, i, o, n, k| {
            note_cpu_kernel(Fp8CpuTier::Scalar);
            crate::gemv_fp8::gemv_fp8_e4m3(b, i, o, n, k)
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx2 => |b, i, o, n, k| {
            note_cpu_kernel(Fp8CpuTier::Avx2);
            unsafe { crate::simd_fp8_avx2::gemv_fp8_e4m3_avx2(b, i, o, n, k) }
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx512 => |b, i, o, n, k| {
            note_cpu_kernel(Fp8CpuTier::Avx512);
            unsafe { crate::simd_fp8_avx512::gemv_fp8_e4m3_avx512(b, i, o, n, k) }
        },
        #[cfg(target_arch = "aarch64")]
        Fp8CpuTier::Neon => |b, i, o, n, k| {
            note_cpu_kernel(Fp8CpuTier::Neon);
            unsafe { crate::simd_fp8_neon::gemv_fp8_e4m3_neon(b, i, o, n, k) }
        },
    }
}

/// See [`dequant_fp8_e4m3_kernel`] for the shared safety argument.
fn gemv_fp8_e5m2_kernel(tier: Fp8CpuTier) -> Fp8GemvFn<BlockFP8E5M2> {
    match tier {
        Fp8CpuTier::Scalar => |b, i, o, n, k| {
            note_cpu_kernel(Fp8CpuTier::Scalar);
            crate::gemv_fp8::gemv_fp8_e5m2(b, i, o, n, k)
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx2 => |b, i, o, n, k| {
            note_cpu_kernel(Fp8CpuTier::Avx2);
            unsafe { crate::simd_fp8_avx2::gemv_fp8_e5m2_avx2(b, i, o, n, k) }
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx512 => |b, i, o, n, k| {
            note_cpu_kernel(Fp8CpuTier::Avx512);
            unsafe { crate::simd_fp8_avx512::gemv_fp8_e5m2_avx512(b, i, o, n, k) }
        },
        #[cfg(target_arch = "aarch64")]
        Fp8CpuTier::Neon => |b, i, o, n, k| {
            note_cpu_kernel(Fp8CpuTier::Neon);
            unsafe { crate::simd_fp8_neon::gemv_fp8_e5m2_neon(b, i, o, n, k) }
        },
    }
}

/// See [`dequant_fp8_e4m3_kernel`] for the shared safety argument.
fn gemm_fp8_e4m3_kernel(tier: Fp8CpuTier) -> Fp8GemmFn<BlockFP8E4M3> {
    match tier {
        Fp8CpuTier::Scalar => |b, i, o, n, k, m| {
            note_cpu_kernel(Fp8CpuTier::Scalar);
            crate::gemm_fp8::gemm_fp8_e4m3(b, i, o, n, k, m)
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx2 => |b, i, o, n, k, m| {
            note_cpu_kernel(Fp8CpuTier::Avx2);
            unsafe { crate::simd_fp8_avx2::gemm_fp8_e4m3_avx2(b, i, o, n, k, m) }
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx512 => |b, i, o, n, k, m| {
            note_cpu_kernel(Fp8CpuTier::Avx512);
            unsafe { crate::simd_fp8_avx512::gemm_fp8_e4m3_avx512(b, i, o, n, k, m) }
        },
        #[cfg(target_arch = "aarch64")]
        Fp8CpuTier::Neon => |b, i, o, n, k, m| {
            note_cpu_kernel(Fp8CpuTier::Neon);
            unsafe { crate::simd_fp8_neon::gemm_fp8_e4m3_neon(b, i, o, n, k, m) }
        },
    }
}

/// See [`dequant_fp8_e4m3_kernel`] for the shared safety argument.
fn gemm_fp8_e5m2_kernel(tier: Fp8CpuTier) -> Fp8GemmFn<BlockFP8E5M2> {
    match tier {
        Fp8CpuTier::Scalar => |b, i, o, n, k, m| {
            note_cpu_kernel(Fp8CpuTier::Scalar);
            crate::gemm_fp8::gemm_fp8_e5m2(b, i, o, n, k, m)
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx2 => |b, i, o, n, k, m| {
            note_cpu_kernel(Fp8CpuTier::Avx2);
            unsafe { crate::simd_fp8_avx2::gemm_fp8_e5m2_avx2(b, i, o, n, k, m) }
        },
        #[cfg(target_arch = "x86_64")]
        Fp8CpuTier::Avx512 => |b, i, o, n, k, m| {
            note_cpu_kernel(Fp8CpuTier::Avx512);
            unsafe { crate::simd_fp8_avx512::gemm_fp8_e5m2_avx512(b, i, o, n, k, m) }
        },
        #[cfg(target_arch = "aarch64")]
        Fp8CpuTier::Neon => |b, i, o, n, k, m| {
            note_cpu_kernel(Fp8CpuTier::Neon);
            unsafe { crate::simd_fp8_neon::gemm_fp8_e5m2_neon(b, i, o, n, k, m) }
        },
    }
}

impl Fp8Kernel for KernelDispatcher {
    /// Dequantize FP8 E4M3FN blocks — tier-aware SIMD dispatch.
    ///
    /// A `KernelTier::Gpu` dispatcher runs the best CPU SIMD tier (K-17): a
    /// per-call GPU dequantization is not worth its transfer, exactly as for
    /// `Q1_0_g128`.
    fn dequant_fp8_e4m3(&self, blocks: &[BlockFP8E4M3], output: &mut [f32]) -> KernelResult<()> {
        dequant_fp8_e4m3_kernel(self.fp8_cpu_tier())(blocks, output)
    }

    /// Dequantize FP8 E5M2 blocks — tier-aware SIMD dispatch.
    ///
    /// A `KernelTier::Gpu` dispatcher runs the best CPU SIMD tier (K-17), as
    /// for [`dequant_fp8_e4m3`](Self::dequant_fp8_e4m3).
    fn dequant_fp8_e5m2(&self, blocks: &[BlockFP8E5M2], output: &mut [f32]) -> KernelResult<()> {
        dequant_fp8_e5m2_kernel(self.fp8_cpu_tier())(blocks, output)
    }

    /// FP8 E4M3FN GEMV — tier-aware SIMD dispatch, with GPU acceleration on
    /// the `KernelTier::Gpu` tier only.
    ///
    /// A CPU-tier dispatcher (`Reference` / `Avx2` / `Avx512` / `Neon`) runs
    /// its own tier's kernel and never enters the GPU path. Dispatch priority
    /// on the `KernelTier::Gpu` path:
    /// 1. Metal (macOS + `metal` feature) — `metal_gemv_fp8_e4m3`.
    /// 2. CUDA (Linux/Windows + `native-cuda` feature) — `cuda_gemv_fp8_e4m3`.
    /// 3. The best CPU SIMD tier, `KernelDispatcher::cpu_tier()` (AVX-512 /
    ///    AVX2 / NEON; the scalar reference only when that is the best this
    ///    CPU has) — K-17 — whenever the build carries no FP8 GPU kernel, the
    ///    GPU path declines the shape, or the kernel fails.
    ///
    /// The raw-byte view of `blocks` the GPU kernels read is sound because
    /// `BlockFP8E4M3` is `#[repr(C)]` with size `BLOCK_FP8_BYTES = 34`.
    fn gemv_fp8_e4m3(
        &self,
        blocks: &[BlockFP8E4M3],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        // `KernelTier::Gpu` only: a CPU-tier dispatcher never enters `gpu`.
        #[cfg(fp8_gpu_gemv)]
        {
            if self.tier() == KernelTier::Gpu && gpu::gemv(blocks, input, output, n_rows, k) {
                return Ok(());
            }
        }
        gemv_fp8_e4m3_kernel(self.fp8_cpu_tier())(blocks, input, output, n_rows, k)
    }

    /// FP8 E5M2 GEMV — tier-aware SIMD dispatch, with GPU acceleration on the
    /// `KernelTier::Gpu` tier only.
    ///
    /// Mirrors [`gemv_fp8_e4m3`](Self::gemv_fp8_e4m3): on `KernelTier::Gpu`,
    /// Metal → CUDA → the best CPU SIMD tier (K-17); on every CPU tier, that
    /// tier's own kernel with no GPU attempt. The raw-byte view is sound
    /// because `BlockFP8E5M2` is `#[repr(C)]` with size `BLOCK_FP8_BYTES = 34`.
    fn gemv_fp8_e5m2(
        &self,
        blocks: &[BlockFP8E5M2],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        // `KernelTier::Gpu` only: a CPU-tier dispatcher never enters `gpu`.
        #[cfg(fp8_gpu_gemv)]
        {
            if self.tier() == KernelTier::Gpu && gpu::gemv(blocks, input, output, n_rows, k) {
                return Ok(());
            }
        }
        gemv_fp8_e5m2_kernel(self.fp8_cpu_tier())(blocks, input, output, n_rows, k)
    }

    /// FP8 E4M3FN GEMM — tier-aware SIMD dispatch.
    ///
    /// A `KernelTier::Gpu` dispatcher runs the best CPU SIMD tier (K-17): the
    /// Metal / CUDA FP8 batch kernels are driven by the model's whole-layer
    /// prefill paths, not through this method.
    fn gemm_fp8_e4m3(
        &self,
        blocks: &[BlockFP8E4M3],
        inputs: &[f32],
        outputs: &mut [f32],
        n_rows: usize,
        k: usize,
        batch: usize,
    ) -> KernelResult<()> {
        gemm_fp8_e4m3_kernel(self.fp8_cpu_tier())(blocks, inputs, outputs, n_rows, k, batch)
    }

    /// FP8 E5M2 GEMM — tier-aware SIMD dispatch.
    ///
    /// A `KernelTier::Gpu` dispatcher runs the best CPU SIMD tier (K-17), as
    /// for [`gemm_fp8_e4m3`](Self::gemm_fp8_e4m3).
    fn gemm_fp8_e5m2(
        &self,
        blocks: &[BlockFP8E5M2],
        inputs: &[f32],
        outputs: &mut [f32],
        n_rows: usize,
        k: usize,
        batch: usize,
    ) -> KernelResult<()> {
        gemm_fp8_e5m2_kernel(self.fp8_cpu_tier())(blocks, inputs, outputs, n_rows, k, batch)
    }

    /// The FP8 kernel family this dispatcher runs.
    ///
    /// A CPU tier reports its own kernels — `fp8_avx512`, `fp8_avx2`,
    /// `fp8_neon` or `fp8_reference`. A `KernelTier::Gpu` dispatcher reports
    /// `fp8_gpu` when this build carries an FP8 GPU GEMV kernel (GEMV on the
    /// GPU; dequantization, GEMM and any GEMV the GPU does not serve on the
    /// CPU SIMD tier), and otherwise the CPU SIMD tier every one of its FP8
    /// calls runs on. (It used to report `fp8_reference` for `Gpu`, the one
    /// kernel family a `Gpu` dispatcher no longer runs unless it genuinely is
    /// the best this CPU has.) Naming routes nothing, so this records no K-17
    /// fallback.
    fn name_fp8(&self) -> &'static str {
        match self.tier() {
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => {
                if FP8_GPU_GEMV_COMPILED {
                    "fp8_gpu"
                } else {
                    fp8_cpu_tier_of(Self::cpu_tier()).name()
                }
            }
            tier => fp8_cpu_tier_of(tier).name(),
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
thread_local! {
    /// FP8 GEMV calls on this thread that entered the GPU path
    /// (`gpu::gemv`) — the observable behind "a CPU-tier dispatcher never
    /// enters the GPU path" (always `0`) and "a `KernelTier::Gpu` dispatcher
    /// tries the GPU kernel this build carries before its K-17 fallback"
    /// (one per call when [`FP8_GPU_GEMV_COMPILED`]). Thread-local, like
    /// `dispatch.rs`'s `LAST_GPU_FALLBACK_TIER`, so concurrently running
    /// tests cannot disturb each other's count.
    static GPU_GEMV_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// FP8 GEMV calls on this thread the GPU kernel *served*: it returned
    /// `Ok`, so `gpu::gemv` handed the dispatcher a finished `output`. With it
    /// a test tells "the GPU produced this result" from "the CPU fallback
    /// did" exactly, instead of inferring it from the numbers (which agree to
    /// rounding) or from a missing K-17 record (which a fallback that skipped
    /// `fp8_cpu_tier` would also leave missing).
    static GPU_GEMV_SERVED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The FP8 GEMV attempt a `KernelTier::Gpu` dispatcher makes before its K-17
/// fallback — compiled only into a build that carries an FP8 GPU GEMV kernel
/// (the `fp8_gpu_gemv` cfg; see [`FP8_GPU_GEMV_COMPILED`]), and reached only
/// from the two `KernelTier::Gpu`-gated call sites in `Fp8Kernel::gemv_fp8_*`
/// above.
#[cfg(fp8_gpu_gemv)]
mod gpu {
    use oxibonsai_core::{BlockFP8E4M3, BlockFP8E5M2, BLOCK_FP8_BYTES, QK_FP8};

    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    use crate::gpu_backend::CudaGraphError;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use crate::gpu_backend::MetalGraphError;

    /// An FP8 block type the GPU GEMV kernels read as raw bytes, bound to the
    /// kernels for its own encoding — so E4M3 blocks can only ever reach the
    /// E4M3 kernel, by type rather than by call-site discipline.
    ///
    /// # Safety
    ///
    /// An implementor must be `#[repr(C)]`, exactly `BLOCK_FP8_BYTES` (34)
    /// bytes long and free of padding — the AoS block
    /// `[q0..q31, scale_lo, scale_hi]` the kernels index — because
    /// [`gemv_operands`] views a `&[Self]` as its initialized bytes.
    pub(super) unsafe trait Fp8GpuBlock: Sized {
        /// The encoding's name in the fallback warning (`"E4M3"` / `"E5M2"`).
        const LABEL: &'static str;

        /// This encoding's Metal GEMV kernel.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        fn metal_gemv(
            bytes: &[u8],
            input: &[f32],
            output: &mut [f32],
            n_rows: usize,
            k: usize,
        ) -> Result<(), MetalGraphError>;

        /// This encoding's CUDA GEMV kernel.
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        fn cuda_gemv(
            bytes: &[u8],
            input: &[f32],
            output: &mut [f32],
            n_rows: usize,
            k: usize,
        ) -> Result<(), CudaGraphError>;
    }

    // SAFETY: `BlockFP8E4M3` is `#[repr(C)] { qs: [u8; 32], d: f16 }` — 32 + 2
    // bytes at alignment 2, so no padding — and exactly `BLOCK_FP8_BYTES`
    // long (const-asserted in the parent module).
    unsafe impl Fp8GpuBlock for BlockFP8E4M3 {
        const LABEL: &'static str = "E4M3";

        #[cfg(all(feature = "metal", target_os = "macos"))]
        fn metal_gemv(
            bytes: &[u8],
            input: &[f32],
            output: &mut [f32],
            n_rows: usize,
            k: usize,
        ) -> Result<(), MetalGraphError> {
            crate::gpu_backend::metal_gemv_fp8_e4m3(bytes, input, output, n_rows, k)
        }

        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        fn cuda_gemv(
            bytes: &[u8],
            input: &[f32],
            output: &mut [f32],
            n_rows: usize,
            k: usize,
        ) -> Result<(), CudaGraphError> {
            crate::gpu_backend::cuda_gemv_fp8_e4m3(bytes, input, output, n_rows, k)
        }
    }

    // SAFETY: `BlockFP8E5M2` has the identical `#[repr(C)] { qs: [u8; 32],
    // d: f16 }` layout — 34 padding-free bytes, const-asserted in the parent
    // module.
    unsafe impl Fp8GpuBlock for BlockFP8E5M2 {
        const LABEL: &'static str = "E5M2";

        #[cfg(all(feature = "metal", target_os = "macos"))]
        fn metal_gemv(
            bytes: &[u8],
            input: &[f32],
            output: &mut [f32],
            n_rows: usize,
            k: usize,
        ) -> Result<(), MetalGraphError> {
            crate::gpu_backend::metal_gemv_fp8_e5m2(bytes, input, output, n_rows, k)
        }

        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        fn cuda_gemv(
            bytes: &[u8],
            input: &[f32],
            output: &mut [f32],
            n_rows: usize,
            k: usize,
        ) -> Result<(), CudaGraphError> {
            crate::gpu_backend::cuda_gemv_fp8_e5m2(bytes, input, output, n_rows, k)
        }
    }

    /// Try the GEMV on this build's GPU kernel — Metal on macOS, CUDA on
    /// Linux/Windows. `true` when the kernel produced `output[..n_rows]`;
    /// `false` when the caller must run its K-17 CPU fallback, because
    /// [`gemv_operands`] declined the call or the kernel failed. A failure
    /// that only means "no device" is silent; any other one is logged.
    pub(super) fn gemv<B: Fp8GpuBlock>(
        blocks: &[B],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> bool {
        #[cfg(test)]
        super::GPU_GEMV_ENTRIES.with(|entries| entries.set(entries.get() + 1));

        let Some(bytes) = gemv_operands(blocks, input.len(), output.len(), n_rows, k) else {
            return false;
        };
        let input = &input[..k];
        let output = &mut output[..n_rows];

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match B::metal_gemv(bytes, input, output, n_rows, k) {
                Ok(()) => return served(),
                Err(e) => {
                    if !is_no_metal_device(&e) {
                        tracing::warn!(
                            error = %e,
                            "Metal FP8 {} GEMV failed, falling back to CPU SIMD",
                            B::LABEL
                        );
                    }
                }
            }
        }

        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        {
            match B::cuda_gemv(bytes, input, output, n_rows, k) {
                Ok(()) => return served(),
                Err(e) => {
                    if !is_no_cuda_device(&e) {
                        tracing::warn!(
                            error = %e,
                            "CUDA FP8 {} GEMV failed, falling back to CPU SIMD",
                            B::LABEL
                        );
                    }
                }
            }
        }

        false
    }

    /// `true` — the GPU kernel served the call — noted under `cfg(test)` in
    /// the parent module's `GPU_GEMV_SERVED`.
    fn served() -> bool {
        #[cfg(test)]
        super::GPU_GEMV_SERVED.with(|count| count.set(count.get() + 1));
        true
    }

    /// How much weight data an `n_rows × k` FP8 GEMV reads.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) struct WeightImage {
        /// `n_rows · (k / QK_FP8)` FP8 blocks.
        pub(super) blocks: usize,
        /// `blocks · BLOCK_FP8_BYTES` bytes.
        pub(super) bytes: usize,
    }

    /// The [`WeightImage`] of an `n_rows × k` FP8 GEMV, or `None` when the
    /// shape alone rules the GPU kernels out. Pure arithmetic on the shape —
    /// no slice is involved — so every limit below is unit-tested directly,
    /// at sizes no test could allocate a weight matrix for. Declined:
    ///
    /// - `n_rows == 0` or `k == 0`: nothing for a launch to compute (a
    ///   zero-sized CUDA grid is itself a launch error);
    /// - `k` not a multiple of `QK_FP8`;
    /// - a block count or byte length that overflows `usize`;
    /// - a weight image of 4 GiB or more: both kernels index it with 32-bit
    ///   byte offsets (`block_idx * 34u`), so it would wrap silently on the
    ///   device instead of failing. (On a 32-bit target the `usize` overflow
    ///   check above already enforces this.)
    ///
    /// That byte bound is the only 32-bit check a shape needs. The kernels
    /// also take `n_rows` and `k` as 32-bit `uint`s, but neither can exceed
    /// one once the image fits: every row holds at least one 34-byte block,
    /// so `n_rows ≤ bytes / 34`; and even a single row's `k / QK_FP8` blocks
    /// take 34 bytes each, so `k / QK_FP8 ≤ bytes / 34`, i.e. `k ≤ 32 ·
    /// bytes / 34 < bytes`. Separate `n_rows` / `k` checks would be
    /// unreachable — no shape could fail one without failing the byte bound
    /// first — so there are none.
    pub(super) fn weight_image(n_rows: usize, k: usize) -> Option<WeightImage> {
        if n_rows == 0 || k == 0 || !k.is_multiple_of(QK_FP8) {
            return None;
        }
        let blocks = n_rows.checked_mul(k / QK_FP8)?;
        let bytes = blocks.checked_mul(BLOCK_FP8_BYTES)?;
        if u32::try_from(bytes).is_err() {
            return None;
        }
        Some(WeightImage { blocks, bytes })
    }

    /// The exact byte image of the `n_rows × (k / QK_FP8)` blocks this GEMV
    /// reads, or `None` when the GPU kernels cannot — or need not — serve the
    /// call and the CPU fallback must: it then also reports the precise error
    /// for a malformed call (`NotBlockAligned` / `DimensionMismatch` /
    /// `BufferTooSmall`), the way `dispatch_std_quant.rs`'s `quant_gpu_bytes`
    /// hands a rejected Q-std shape to its CPU fallback. Declined: every shape
    /// [`weight_image`] declines — before `blocks` is consulted at all — and
    /// any call with a buffer shorter than it reads.
    ///
    /// The image is trimmed to exactly what the call reads — as the caller
    /// trims `input` / `output` to `k` / `n_rows` — because the Metal wrapper
    /// rejects buffers longer than `n_rows` / `k` imply, which the CPU
    /// kernels accept.
    pub(super) fn gemv_operands<B: Fp8GpuBlock>(
        blocks: &[B],
        input_len: usize,
        output_len: usize,
        n_rows: usize,
        k: usize,
    ) -> Option<&[u8]> {
        const { assert!(std::mem::size_of::<B>() == BLOCK_FP8_BYTES) };
        let image = weight_image(n_rows, k)?;
        if blocks.len() < image.blocks || input_len < k || output_len < n_rows {
            return None;
        }
        // SAFETY: `B: Fp8GpuBlock` guarantees `#[repr(C)]`, padding-free,
        // `BLOCK_FP8_BYTES`-sized blocks (also checked at compile time just
        // above), so the leading `image.blocks` blocks — `image.blocks <=
        // blocks.len()`, checked above — are exactly `image.bytes` (`=
        // image.blocks * BLOCK_FP8_BYTES`, overflow-checked by
        // `weight_image`) contiguous, initialized bytes. The view borrows
        // `blocks` and cannot outlive it.
        Some(unsafe { std::slice::from_raw_parts(blocks.as_ptr().cast::<u8>(), image.bytes) })
    }

    /// Whether a Metal FP8 GEMV failure only means this host has no Metal
    /// device — the expected state of a GPU-less machine, so the fallback
    /// stays silent. Matched by variant: no allocation, and no dependency on
    /// the error's `Display` text.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn is_no_metal_device(e: &MetalGraphError) -> bool {
        matches!(e, MetalGraphError::DeviceNotFound)
    }

    /// Whether a CUDA FP8 GEMV failure only means this host has no CUDA
    /// device (or driver) — the expected state of a GPU-less machine, so the
    /// fallback stays silent. Matched by variant, like `is_no_metal_device`.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    pub(super) fn is_no_cuda_device(e: &CudaGraphError) -> bool {
        matches!(e, CudaGraphError::DeviceNotFound(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "gpu")]
    use crate::dispatch::LAST_GPU_FALLBACK_TIER;
    #[cfg(feature = "gpu")]
    use crate::error::KernelError;
    use oxibonsai_core::QK_FP8;

    // ── Fixtures ─────────────────────────────────────────────────────────

    /// Deterministic values in `[-scale, scale)` from an xorshift stream, so
    /// every block gets its own codes and scale.
    fn signed_stream(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            })
            .collect()
    }

    /// One FP8 problem in both encodings: an `n_rows × k` weight matrix and
    /// `batch` input rows of width `k` (the GEMVs use the first one).
    struct Fixture {
        n_rows: usize,
        k: usize,
        batch: usize,
        e4m3: Vec<BlockFP8E4M3>,
        e5m2: Vec<BlockFP8E5M2>,
        inputs: Vec<f32>,
    }

    fn fixture() -> Fixture {
        let (n_rows, k, batch) = (5, 3 * QK_FP8, 3);
        let weights = signed_stream(n_rows * k, 0xF8D1_5BA7, 0.75);
        Fixture {
            n_rows,
            k,
            batch,
            e4m3: BlockFP8E4M3::quantize(&weights).expect("quantize e4m3 fixture"),
            e5m2: BlockFP8E5M2::quantize(&weights).expect("quantize e5m2 fixture"),
            inputs: signed_stream(batch * k, 0x1A7E_57ED, 1.5),
        }
    }

    /// Every CPU tier this architecture defines.
    #[cfg(target_arch = "x86_64")]
    const CPU_TIERS: &[KernelTier] = &[KernelTier::Reference, KernelTier::Avx2, KernelTier::Avx512];
    /// Every CPU tier this architecture defines.
    #[cfg(target_arch = "aarch64")]
    const CPU_TIERS: &[KernelTier] = &[KernelTier::Reference, KernelTier::Neon];
    /// Every CPU tier this architecture defines.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    const CPU_TIERS: &[KernelTier] = &[KernelTier::Reference];

    /// A dispatcher for every CPU tier `with_tier` can hand out on this host
    /// (an x86 tier the CPU lacks is clamped down by `with_tier` itself, so
    /// requesting all of them is safe on every machine).
    fn cpu_dispatchers() -> Vec<KernelDispatcher> {
        CPU_TIERS
            .iter()
            .map(|&tier| KernelDispatcher::with_tier(tier))
            .collect()
    }

    fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}[{i}]: dispatcher produced {g}, the tier's own kernel {w}"
            );
        }
    }

    #[cfg(feature = "gpu")]
    fn reset_gpu_observers() {
        LAST_GPU_FALLBACK_TIER.with(|c| c.set(None));
        GPU_GEMV_ENTRIES.with(|c| c.set(0));
        GPU_GEMV_SERVED.with(|c| c.set(0));
        LAST_FP8_CPU_KERNEL.with(|c| c.set(None));
    }

    #[cfg(feature = "gpu")]
    fn last_fallback() -> Option<KernelTier> {
        LAST_GPU_FALLBACK_TIER.with(std::cell::Cell::get)
    }

    #[cfg(feature = "gpu")]
    fn gpu_gemv_entries() -> usize {
        GPU_GEMV_ENTRIES.with(std::cell::Cell::get)
    }

    #[cfg(feature = "gpu")]
    fn gpu_gemv_served() -> usize {
        GPU_GEMV_SERVED.with(std::cell::Cell::get)
    }

    /// The FP8 CPU kernel family the last dequant / GEMV / GEMM on this
    /// thread ran (`LAST_FP8_CPU_KERNEL`) — cleared as it is read, so every
    /// call checked must note its own kernel rather than inherit a note.
    fn take_cpu_kernel() -> Option<Fp8CpuTier> {
        LAST_FP8_CPU_KERNEL.with(std::cell::Cell::take)
    }

    /// The FP8 CPU kernel family each CPU [`KernelTier`] must run — spelled
    /// out here, exhaustively, rather than read from `fp8_cpu_tier_of`, so a
    /// drift on either side is caught and a new tier has to be decided here
    /// too.
    fn expected_fp8_cpu_tier(tier: KernelTier) -> Fp8CpuTier {
        match tier {
            KernelTier::Reference => Fp8CpuTier::Scalar,
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => Fp8CpuTier::Avx2,
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => Fp8CpuTier::Avx512,
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => Fp8CpuTier::Neon,
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => unreachable!("`Gpu` is not a CPU tier"),
        }
    }

    // ── Oracles: the kernel each CPU tier must run, called directly ──────
    //
    // Deliberately independent of this module's selectors, so a selector that
    // routes a tier to the wrong kernel cannot also bend the expectation.
    //
    // SAFETY (every `unsafe` call below): `tier` is always a `with_tier`
    // dispatcher's tier, which `clamp_tier_to_cpu` re-validated against this
    // CPU, or `KernelDispatcher::cpu_tier()`, which detects it at runtime;
    // NEON is the AArch64 baseline.

    fn oracle_dequant_e4m3(
        tier: KernelTier,
        blocks: &[BlockFP8E4M3],
        output: &mut [f32],
    ) -> KernelResult<()> {
        match tier {
            KernelTier::Reference => crate::dequant_fp8::dequant_fp8_e4m3(blocks, output),
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::dequant_fp8_e4m3_avx2(blocks, output)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::dequant_fp8_e4m3_avx512(blocks, output)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::dequant_fp8_e4m3_neon(blocks, output)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => unreachable!("the oracles cover CPU tiers only"),
        }
    }

    fn oracle_dequant_e5m2(
        tier: KernelTier,
        blocks: &[BlockFP8E5M2],
        output: &mut [f32],
    ) -> KernelResult<()> {
        match tier {
            KernelTier::Reference => crate::dequant_fp8::dequant_fp8_e5m2(blocks, output),
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::dequant_fp8_e5m2_avx2(blocks, output)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::dequant_fp8_e5m2_avx512(blocks, output)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::dequant_fp8_e5m2_neon(blocks, output)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => unreachable!("the oracles cover CPU tiers only"),
        }
    }

    fn oracle_gemv_e4m3(
        tier: KernelTier,
        blocks: &[BlockFP8E4M3],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match tier {
            KernelTier::Reference => {
                crate::gemv_fp8::gemv_fp8_e4m3(blocks, input, output, n_rows, k)
            }
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::gemv_fp8_e4m3_avx2(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::gemv_fp8_e4m3_avx512(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::gemv_fp8_e4m3_neon(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => unreachable!("the oracles cover CPU tiers only"),
        }
    }

    fn oracle_gemv_e5m2(
        tier: KernelTier,
        blocks: &[BlockFP8E5M2],
        input: &[f32],
        output: &mut [f32],
        n_rows: usize,
        k: usize,
    ) -> KernelResult<()> {
        match tier {
            KernelTier::Reference => {
                crate::gemv_fp8::gemv_fp8_e5m2(blocks, input, output, n_rows, k)
            }
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::gemv_fp8_e5m2_avx2(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::gemv_fp8_e5m2_avx512(blocks, input, output, n_rows, k)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::gemv_fp8_e5m2_neon(blocks, input, output, n_rows, k)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => unreachable!("the oracles cover CPU tiers only"),
        }
    }

    fn oracle_gemm_e4m3(
        tier: KernelTier,
        blocks: &[BlockFP8E4M3],
        inputs: &[f32],
        outputs: &mut [f32],
        f: &Fixture,
    ) -> KernelResult<()> {
        let (n, k, m) = (f.n_rows, f.k, f.batch);
        match tier {
            KernelTier::Reference => {
                crate::gemm_fp8::gemm_fp8_e4m3(blocks, inputs, outputs, n, k, m)
            }
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::gemm_fp8_e4m3_avx2(blocks, inputs, outputs, n, k, m)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::gemm_fp8_e4m3_avx512(blocks, inputs, outputs, n, k, m)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::gemm_fp8_e4m3_neon(blocks, inputs, outputs, n, k, m)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => unreachable!("the oracles cover CPU tiers only"),
        }
    }

    fn oracle_gemm_e5m2(
        tier: KernelTier,
        blocks: &[BlockFP8E5M2],
        inputs: &[f32],
        outputs: &mut [f32],
        f: &Fixture,
    ) -> KernelResult<()> {
        let (n, k, m) = (f.n_rows, f.k, f.batch);
        match tier {
            KernelTier::Reference => {
                crate::gemm_fp8::gemm_fp8_e5m2(blocks, inputs, outputs, n, k, m)
            }
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => unsafe {
                crate::simd_fp8_avx2::gemm_fp8_e5m2_avx2(blocks, inputs, outputs, n, k, m)
            },
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => unsafe {
                crate::simd_fp8_avx512::gemm_fp8_e5m2_avx512(blocks, inputs, outputs, n, k, m)
            },
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => unsafe {
                crate::simd_fp8_neon::gemm_fp8_e5m2_neon(blocks, inputs, outputs, n, k, m)
            },
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => unreachable!("the oracles cover CPU tiers only"),
        }
    }

    /// The `name_fp8` label a CPU tier's kernels carry — spelled out here
    /// rather than read from `Fp8CpuTier::name`, so a drift in either side is
    /// caught.
    fn expected_cpu_label(tier: KernelTier) -> &'static str {
        match tier {
            KernelTier::Reference => "fp8_reference",
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx2 => "fp8_avx2",
            #[cfg(target_arch = "x86_64")]
            KernelTier::Avx512 => "fp8_avx512",
            #[cfg(target_arch = "aarch64")]
            KernelTier::Neon => "fp8_neon",
            #[cfg(feature = "gpu")]
            KernelTier::Gpu => unreachable!("`Gpu` is not a CPU tier"),
        }
    }

    // ── The tier mapping itself ──────────────────────────────────────────

    /// `fp8_cpu_tier_of` sends every [`KernelTier`] to its own kernel family
    /// — the mapping behind every dispatcher's dequant, whose outputs cannot
    /// tell the families apart — and `Gpu`, which `fp8_cpu_tier` resolves to
    /// `cpu_tier()` before mapping, to the scalar reference only as the
    /// documented defensive default that keeps the function total.
    #[test]
    fn fp8_cpu_tier_of_maps_every_kernel_tier_onto_its_own_kernel_family() {
        assert_eq!(fp8_cpu_tier_of(KernelTier::Reference), Fp8CpuTier::Scalar);
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(fp8_cpu_tier_of(KernelTier::Avx2), Fp8CpuTier::Avx2);
            assert_eq!(fp8_cpu_tier_of(KernelTier::Avx512), Fp8CpuTier::Avx512);
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(fp8_cpu_tier_of(KernelTier::Neon), Fp8CpuTier::Neon);
        #[cfg(feature = "gpu")]
        assert_eq!(fp8_cpu_tier_of(KernelTier::Gpu), Fp8CpuTier::Scalar);

        // Every CPU tier this architecture defines, against the exhaustive
        // oracle the routing tests below rely on, and with the label
        // `name_fp8` reports for it.
        for &tier in CPU_TIERS {
            let family = fp8_cpu_tier_of(tier);
            assert_eq!(family, expected_fp8_cpu_tier(tier), "{tier:?}");
            assert_eq!(family.name(), expected_cpu_label(tier), "{tier:?}");
        }
    }

    // ── CPU tiers: their own kernel, never the GPU ───────────────────────

    /// The defect this module's tier gate fixes: a CPU-tier dispatcher's FP8
    /// GEMV must run exactly its own tier's kernel — the kernel that ran must
    /// have noted that tier's family, and the result must match that kernel
    /// bit for bit, so a result computed anywhere else (the CUDA kernel, which
    /// on a GPU host is what used to produce it) cannot pass — and must
    /// neither enter the GPU path nor record a GPU fallback.
    #[test]
    fn cpu_tier_fp8_gemv_never_enters_the_gpu_path() {
        let f = fixture();
        let input = &f.inputs[..f.k];
        for dispatcher in cpu_dispatchers() {
            let tier = dispatcher.tier();
            let family = Some(expected_fp8_cpu_tier(tier));
            #[cfg(feature = "gpu")]
            reset_gpu_observers();

            let mut got = vec![0.0f32; f.n_rows];
            let mut want = vec![0.0f32; f.n_rows];
            dispatcher
                .gemv_fp8_e4m3(&f.e4m3, input, &mut got, f.n_rows, f.k)
                .expect("dispatcher FP8 E4M3 GEMV");
            assert_eq!(take_cpu_kernel(), family, "{tier:?} gemv_fp8_e4m3 ran");
            oracle_gemv_e4m3(tier, &f.e4m3, input, &mut want, f.n_rows, f.k)
                .expect("tier's own FP8 E4M3 GEMV");
            assert!(want.iter().all(|v| v.is_finite()), "fixture must be finite");
            assert_bits_eq(&got, &want, &format!("{tier:?} gemv_fp8_e4m3"));

            let mut got = vec![0.0f32; f.n_rows];
            let mut want = vec![0.0f32; f.n_rows];
            dispatcher
                .gemv_fp8_e5m2(&f.e5m2, input, &mut got, f.n_rows, f.k)
                .expect("dispatcher FP8 E5M2 GEMV");
            assert_eq!(take_cpu_kernel(), family, "{tier:?} gemv_fp8_e5m2 ran");
            oracle_gemv_e5m2(tier, &f.e5m2, input, &mut want, f.n_rows, f.k)
                .expect("tier's own FP8 E5M2 GEMV");
            assert_bits_eq(&got, &want, &format!("{tier:?} gemv_fp8_e5m2"));

            #[cfg(feature = "gpu")]
            {
                assert_eq!(
                    gpu_gemv_entries(),
                    0,
                    "a {tier:?} dispatcher's FP8 GEMV entered the GPU path"
                );
                assert_eq!(
                    last_fallback(),
                    None,
                    "a {tier:?} dispatcher's FP8 GEMV recorded a GPU fallback"
                );
            }
        }
    }

    /// The selector refactor must leave every CPU tier's dequant and GEMM on
    /// exactly the kernel it ran before, with no GPU routing. Bit equality
    /// with the tier's own kernel pins GEMM, but not dequantization — every
    /// tier's dequant agrees bit for bit (one exact decode times the block
    /// scale per element), so the scalar one would pass for any tier — which
    /// is why each call must also have noted the tier's kernel family as the
    /// one that ran.
    #[test]
    fn cpu_tier_fp8_dequant_and_gemm_run_their_own_tier_kernel() {
        let f = fixture();
        for dispatcher in cpu_dispatchers() {
            let tier = dispatcher.tier();
            let family = Some(expected_fp8_cpu_tier(tier));
            #[cfg(feature = "gpu")]
            reset_gpu_observers();

            let mut got = vec![0.0f32; f.e4m3.len() * QK_FP8];
            let mut want = vec![0.0f32; f.e4m3.len() * QK_FP8];
            dispatcher
                .dequant_fp8_e4m3(&f.e4m3, &mut got)
                .expect("dispatcher dequant e4m3");
            assert_eq!(take_cpu_kernel(), family, "{tier:?} dequant_fp8_e4m3 ran");
            oracle_dequant_e4m3(tier, &f.e4m3, &mut want).expect("tier dequant e4m3");
            assert_bits_eq(&got, &want, &format!("{tier:?} dequant_fp8_e4m3"));

            let mut got = vec![0.0f32; f.e5m2.len() * QK_FP8];
            let mut want = vec![0.0f32; f.e5m2.len() * QK_FP8];
            dispatcher
                .dequant_fp8_e5m2(&f.e5m2, &mut got)
                .expect("dispatcher dequant e5m2");
            assert_eq!(take_cpu_kernel(), family, "{tier:?} dequant_fp8_e5m2 ran");
            oracle_dequant_e5m2(tier, &f.e5m2, &mut want).expect("tier dequant e5m2");
            assert_bits_eq(&got, &want, &format!("{tier:?} dequant_fp8_e5m2"));

            let mut got = vec![0.0f32; f.batch * f.n_rows];
            let mut want = vec![0.0f32; f.batch * f.n_rows];
            dispatcher
                .gemm_fp8_e4m3(&f.e4m3, &f.inputs, &mut got, f.n_rows, f.k, f.batch)
                .expect("dispatcher gemm e4m3");
            assert_eq!(take_cpu_kernel(), family, "{tier:?} gemm_fp8_e4m3 ran");
            oracle_gemm_e4m3(tier, &f.e4m3, &f.inputs, &mut want, &f).expect("tier gemm e4m3");
            assert_bits_eq(&got, &want, &format!("{tier:?} gemm_fp8_e4m3"));

            let mut got = vec![0.0f32; f.batch * f.n_rows];
            let mut want = vec![0.0f32; f.batch * f.n_rows];
            dispatcher
                .gemm_fp8_e5m2(&f.e5m2, &f.inputs, &mut got, f.n_rows, f.k, f.batch)
                .expect("dispatcher gemm e5m2");
            assert_eq!(take_cpu_kernel(), family, "{tier:?} gemm_fp8_e5m2 ran");
            oracle_gemm_e5m2(tier, &f.e5m2, &f.inputs, &mut want, &f).expect("tier gemm e5m2");
            assert_bits_eq(&got, &want, &format!("{tier:?} gemm_fp8_e5m2"));

            #[cfg(feature = "gpu")]
            {
                assert_eq!(gpu_gemv_entries(), 0, "{tier:?}: no GPU path here");
                assert_eq!(last_fallback(), None, "{tier:?}: no GPU fallback here");
            }
        }
    }

    // ── KernelTier::Gpu: GPU first, then the CPU SIMD tier (K-17) ───────

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_gemv_fp8_e4m3_routes_through_cpu_simd_fallback() {
        reset_gpu_observers();
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let blocks: Vec<BlockFP8E4M3> = Vec::new();
        // `k = 31` is not a multiple of `QK_FP8` (32), so `gpu::gemv_operands`
        // declines the call before any device work — deterministic on every
        // machine, with or without a GPU (the same trick
        // `gpu_tier_gemv_q4_0_routes_through_cpu_simd_fallback` uses).
        let input = vec![0.0f32; 31];
        let mut output = vec![0.0f32; 1];

        let result = dispatcher.gemv_fp8_e4m3(&blocks, &input, &mut output, 1, 31);

        // The CPU tier, not the GPU wrapper, reports the precise error.
        assert!(
            matches!(
                result,
                Err(KernelError::NotBlockAligned {
                    count: 31,
                    block_size: QK_FP8
                })
            ),
            "{result:?}"
        );
        assert_eq!(
            last_fallback(),
            Some(KernelDispatcher::cpu_tier()),
            "KernelTier::Gpu arm of gemv_fp8_e4m3 must route through the best CPU SIMD \
             tier (K-17), not call the scalar kernel directly"
        );
        assert_eq!(
            take_cpu_kernel(),
            Some(expected_fp8_cpu_tier(KernelDispatcher::cpu_tier())),
            "the K-17 fallback of gemv_fp8_e4m3 must run the CPU SIMD tier's own kernel"
        );
        assert_eq!(
            gpu_gemv_entries(),
            usize::from(FP8_GPU_GEMV_COMPILED),
            "a Gpu-tier dispatcher must try the FP8 GPU GEMV kernel this build carries \
             (and only then fall back)"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_gemv_fp8_e5m2_routes_through_cpu_simd_fallback() {
        reset_gpu_observers();
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let blocks: Vec<BlockFP8E5M2> = Vec::new();
        // See the E4M3 twin above: `k = 31` is declined before any device work.
        let input = vec![0.0f32; 31];
        let mut output = vec![0.0f32; 1];

        let result = dispatcher.gemv_fp8_e5m2(&blocks, &input, &mut output, 1, 31);

        assert!(
            matches!(
                result,
                Err(KernelError::NotBlockAligned {
                    count: 31,
                    block_size: QK_FP8
                })
            ),
            "{result:?}"
        );
        assert_eq!(
            last_fallback(),
            Some(KernelDispatcher::cpu_tier()),
            "KernelTier::Gpu arm of gemv_fp8_e5m2 must route through the best CPU SIMD \
             tier (K-17), not call the scalar kernel directly"
        );
        assert_eq!(
            take_cpu_kernel(),
            Some(expected_fp8_cpu_tier(KernelDispatcher::cpu_tier())),
            "the K-17 fallback of gemv_fp8_e5m2 must run the CPU SIMD tier's own kernel"
        );
        assert_eq!(
            gpu_gemv_entries(),
            usize::from(FP8_GPU_GEMV_COMPILED),
            "a Gpu-tier dispatcher must try the FP8 GPU GEMV kernel this build carries \
             (and only then fall back)"
        );
    }

    /// A *valid* call the GPU path declines — no rows, nothing to launch —
    /// still succeeds, through the same K-17 routing.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_fp8_gemv_with_no_rows_succeeds_on_the_cpu_simd_tier() {
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let input = vec![0.5f32; QK_FP8];
        let cpu = KernelDispatcher::cpu_tier();

        reset_gpu_observers();
        dispatcher
            .gemv_fp8_e4m3(&[], &input, &mut [], 0, QK_FP8)
            .expect("a zero-row E4M3 GEMV is a valid no-op");
        assert_eq!(last_fallback(), Some(cpu));
        assert_eq!(take_cpu_kernel(), Some(expected_fp8_cpu_tier(cpu)));
        assert_eq!(gpu_gemv_entries(), usize::from(FP8_GPU_GEMV_COMPILED));
        assert_eq!(gpu_gemv_served(), 0, "nothing to launch");

        reset_gpu_observers();
        dispatcher
            .gemv_fp8_e5m2(&[], &input, &mut [], 0, QK_FP8)
            .expect("a zero-row E5M2 GEMV is a valid no-op");
        assert_eq!(last_fallback(), Some(cpu));
        assert_eq!(take_cpu_kernel(), Some(expected_fp8_cpu_tier(cpu)));
        assert_eq!(gpu_gemv_entries(), usize::from(FP8_GPU_GEMV_COMPILED));
        assert_eq!(gpu_gemv_served(), 0, "nothing to launch");
    }

    /// `|got - want| <= 1e-4 + 1e-4 · max(|got|, |want|, 1)` element-wise —
    /// the GPU kernels sum in a different order than the CPU ones.
    #[cfg(feature = "gpu")]
    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let tol = 1e-4 + 1e-4 * g.abs().max(w.abs()).max(1.0);
            assert!(
                g.is_finite() && (g - w).abs() <= tol,
                "{what}[{i}]: GPU {g} vs CPU SIMD tier {w} (tolerance {tol})"
            );
        }
    }

    /// Whether this process holds a live context on the GPU this build's FP8
    /// GEMV kernel runs on — device state, read without opening a device and
    /// independently of this module's own instrumentation. The GPU can only
    /// have served a call if this holds once the call has returned.
    ///
    /// CUDA: every CUDA FP8 GEMV goes through `CudaGraph::global()`, whose
    /// state `cuda_context_is_live` mirrors lock-free; it reaches `READY` only
    /// when a context was created, and never leaves it.
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    fn gpu_context_is_live() -> bool {
        crate::gpu_backend::cuda_graph::functions::cuda_context_is_live()
    }

    /// Metal: every Metal FP8 GEMV dispatches into `MetalGraph::global()` —
    /// this thread's bound session, else the process-default session, which
    /// once created stays alive on the shared device for the life of the
    /// process — so a served call leaves one of these two true. It is the
    /// "is a Metal device live, without opening one" test
    /// `scirs2_weight_cache::release_metal_mirrors` makes.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn gpu_context_is_live() -> bool {
        use crate::gpu_backend::MetalGraph;
        MetalGraph::current_session().is_some() || MetalGraph::live_session_count() > 0
    }

    /// No FP8 GPU GEMV kernel in this build, so no GPU can serve one.
    #[cfg(all(feature = "gpu", not(fp8_gpu_gemv)))]
    fn gpu_context_is_live() -> bool {
        false
    }

    /// The full K-17 contract for a well-formed call on `KernelTier::Gpu`:
    /// exactly one of two outcomes, and the test can tell which.
    ///
    /// - The GPU kernel this build carries served it: `gpu::gemv` reported it
    ///   served (`GPU_GEMV_SERVED`) *and* a GPU context is live after the
    ///   call (`gpu_context_is_live`, device state rather than anything this
    ///   module records); no K-17 fallback was recorded and no CPU kernel
    ///   ran; and the result agrees with the CPU SIMD tier to summation-order
    ///   rounding.
    /// - Otherwise it must have fallen back: routed through `fp8_cpu_tier`
    ///   (recorded), run on the best CPU SIMD tier's own kernel (noted), and
    ///   bit-identical to that kernel's result.
    ///
    /// With no live context — no device (`CUDA_VISIBLE_DEVICES=""`), or no
    /// FP8 GPU kernel in this build — the fallback is therefore *required*:
    /// a `Gpu` arm that regressed to the scalar reference without the K-17
    /// routing fails here instead of passing as "served by the GPU".
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_fp8_gemv_is_served_by_the_gpu_kernel_or_the_cpu_simd_tier() {
        // Serialize against the crate's other tests that drive the CUDA
        // singleton's one stream (see `gpu_parity_test_guard`).
        #[cfg(all(
            feature = "native-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        let _serial = crate::gpu_backend::cuda_graph::types::gpu_parity_test_guard();

        let f = fixture();
        let input = &f.inputs[..f.k];
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let cpu = KernelDispatcher::cpu_tier();
        let check = |got: &[f32], want: &[f32], what: &str| {
            let (routed, ran, served) = (last_fallback(), take_cpu_kernel(), gpu_gemv_served());
            assert_eq!(
                gpu_gemv_entries(),
                usize::from(FP8_GPU_GEMV_COMPILED),
                "{what}: a Gpu-tier dispatcher tries the FP8 GPU GEMV kernel this build \
                 carries, once"
            );
            if served == 1 && gpu_context_is_live() {
                assert_eq!(
                    routed, None,
                    "{what}: served by the GPU, yet routed to the CPU"
                );
                assert_eq!(
                    ran, None,
                    "{what}: served by the GPU, yet a CPU kernel ran too"
                );
                assert_close(got, want, what);
            } else {
                assert_eq!(
                    served, 0,
                    "{what}: reported served by the GPU with no live GPU context"
                );
                assert_eq!(
                    routed,
                    Some(cpu),
                    "{what}: not served by the GPU, so it must take the K-17 route to the \
                     best CPU SIMD tier"
                );
                assert_eq!(
                    ran,
                    Some(expected_fp8_cpu_tier(cpu)),
                    "{what}: the K-17 fallback must run the CPU SIMD tier's own kernel"
                );
                assert_bits_eq(got, want, what);
            }
        };

        reset_gpu_observers();
        let mut got = vec![0.0f32; f.n_rows];
        let mut want = vec![0.0f32; f.n_rows];
        dispatcher
            .gemv_fp8_e4m3(&f.e4m3, input, &mut got, f.n_rows, f.k)
            .expect("Gpu-tier FP8 E4M3 GEMV");
        oracle_gemv_e4m3(cpu, &f.e4m3, input, &mut want, f.n_rows, f.k)
            .expect("CPU SIMD tier FP8 E4M3 GEMV");
        check(&got, &want, "Gpu-tier gemv_fp8_e4m3");

        reset_gpu_observers();
        let mut got = vec![0.0f32; f.n_rows];
        let mut want = vec![0.0f32; f.n_rows];
        dispatcher
            .gemv_fp8_e5m2(&f.e5m2, input, &mut got, f.n_rows, f.k)
            .expect("Gpu-tier FP8 E5M2 GEMV");
        oracle_gemv_e5m2(cpu, &f.e5m2, input, &mut want, f.n_rows, f.k)
            .expect("CPU SIMD tier FP8 E5M2 GEMV");
        check(&got, &want, "Gpu-tier gemv_fp8_e5m2");
    }

    /// Audit result for dequant / GEMM: no GPU path to gate, but their `Gpu`
    /// arm used to fall through `_ =>` to the scalar reference. It must run
    /// the best CPU SIMD tier instead (K-17) — that tier's own kernel, noted
    /// as the one that ran (the only thing that tells dequantization's tiers
    /// apart) and bit-identical to it — and record the routing like every
    /// other K-17 fallback.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_fp8_dequant_and_gemm_route_through_the_cpu_simd_tier() {
        let f = fixture();
        let dispatcher = KernelDispatcher::with_tier(KernelTier::Gpu);
        let cpu = KernelDispatcher::cpu_tier();
        let family = Some(expected_fp8_cpu_tier(cpu));

        reset_gpu_observers();
        let mut got = vec![0.0f32; f.e4m3.len() * QK_FP8];
        let mut want = vec![0.0f32; f.e4m3.len() * QK_FP8];
        dispatcher
            .dequant_fp8_e4m3(&f.e4m3, &mut got)
            .expect("Gpu-tier dequant e4m3");
        assert_eq!(take_cpu_kernel(), family, "Gpu-tier dequant_fp8_e4m3 ran");
        oracle_dequant_e4m3(cpu, &f.e4m3, &mut want).expect("CPU SIMD dequant e4m3");
        assert_bits_eq(&got, &want, "Gpu-tier dequant_fp8_e4m3");
        assert_eq!(last_fallback(), Some(cpu), "dequant_fp8_e4m3 (K-17)");

        reset_gpu_observers();
        let mut got = vec![0.0f32; f.e5m2.len() * QK_FP8];
        let mut want = vec![0.0f32; f.e5m2.len() * QK_FP8];
        dispatcher
            .dequant_fp8_e5m2(&f.e5m2, &mut got)
            .expect("Gpu-tier dequant e5m2");
        assert_eq!(take_cpu_kernel(), family, "Gpu-tier dequant_fp8_e5m2 ran");
        oracle_dequant_e5m2(cpu, &f.e5m2, &mut want).expect("CPU SIMD dequant e5m2");
        assert_bits_eq(&got, &want, "Gpu-tier dequant_fp8_e5m2");
        assert_eq!(last_fallback(), Some(cpu), "dequant_fp8_e5m2 (K-17)");

        reset_gpu_observers();
        let mut got = vec![0.0f32; f.batch * f.n_rows];
        let mut want = vec![0.0f32; f.batch * f.n_rows];
        dispatcher
            .gemm_fp8_e4m3(&f.e4m3, &f.inputs, &mut got, f.n_rows, f.k, f.batch)
            .expect("Gpu-tier gemm e4m3");
        assert_eq!(take_cpu_kernel(), family, "Gpu-tier gemm_fp8_e4m3 ran");
        oracle_gemm_e4m3(cpu, &f.e4m3, &f.inputs, &mut want, &f).expect("CPU SIMD gemm e4m3");
        assert_bits_eq(&got, &want, "Gpu-tier gemm_fp8_e4m3");
        assert_eq!(last_fallback(), Some(cpu), "gemm_fp8_e4m3 (K-17)");

        reset_gpu_observers();
        let mut got = vec![0.0f32; f.batch * f.n_rows];
        let mut want = vec![0.0f32; f.batch * f.n_rows];
        dispatcher
            .gemm_fp8_e5m2(&f.e5m2, &f.inputs, &mut got, f.n_rows, f.k, f.batch)
            .expect("Gpu-tier gemm e5m2");
        assert_eq!(take_cpu_kernel(), family, "Gpu-tier gemm_fp8_e5m2 ran");
        oracle_gemm_e5m2(cpu, &f.e5m2, &f.inputs, &mut want, &f).expect("CPU SIMD gemm e5m2");
        assert_bits_eq(&got, &want, "Gpu-tier gemm_fp8_e5m2");
        assert_eq!(last_fallback(), Some(cpu), "gemm_fp8_e5m2 (K-17)");

        // Neither has a per-call GPU kernel, so neither entered the GPU path.
        assert_eq!(gpu_gemv_entries(), 0);
    }

    #[test]
    fn name_fp8_names_the_kernel_family_each_tier_runs() {
        for dispatcher in cpu_dispatchers() {
            assert_eq!(
                dispatcher.name_fp8(),
                expected_cpu_label(dispatcher.tier()),
                "{:?}",
                dispatcher.tier()
            );
        }
        #[cfg(feature = "gpu")]
        {
            reset_gpu_observers();
            let gpu = KernelDispatcher::with_tier(KernelTier::Gpu);
            let want = if FP8_GPU_GEMV_COMPILED {
                "fp8_gpu"
            } else {
                expected_cpu_label(KernelDispatcher::cpu_tier())
            };
            assert_eq!(gpu.name_fp8(), want);
            assert_eq!(
                last_fallback(),
                None,
                "naming must not record a K-17 fallback"
            );
        }
    }

    // ── The GPU path's own pieces ────────────────────────────────────────

    #[cfg(fp8_gpu_gemv)]
    #[test]
    fn gpu_gemv_operands_are_exactly_what_the_kernels_read() {
        use oxibonsai_core::BLOCK_FP8_BYTES;

        let f = fixture();
        let (n_rows, k) = (f.n_rows, f.k);
        let row_bytes = (k / QK_FP8) * BLOCK_FP8_BYTES;

        // Oversized `input` / `output` are fine (the CPU kernels accept them
        // and the caller trims both); the image is exactly `n_rows` rows —
        // the size `gpu::weight_image` computes for the shape.
        let bytes = gpu::gemv_operands(&f.e4m3, k + 7, n_rows + 3, n_rows, k)
            .expect("a well-formed shape is GPU-eligible");
        assert_eq!(bytes.len(), n_rows * row_bytes);
        assert_eq!(
            Some(bytes.len()),
            gpu::weight_image(n_rows, k).map(|image| image.bytes)
        );
        assert_eq!(bytes.as_ptr(), f.e4m3.as_ptr().cast::<u8>());
        // Fewer rows than `blocks` holds: only the leading rows are handed over.
        let first = gpu::gemv_operands(&f.e5m2, k, 1, 1, k).expect("one row is eligible");
        assert_eq!(first.len(), row_bytes);

        // Declined, leaving the CPU tier to compute (or to report the error):
        assert!(
            gpu::gemv_operands(&f.e4m3, k, n_rows, 0, k).is_none(),
            "no rows"
        );
        assert!(
            gpu::gemv_operands(&f.e4m3, k, n_rows, n_rows, 0).is_none(),
            "k = 0"
        );
        assert!(
            gpu::gemv_operands(&f.e4m3, k, n_rows, n_rows, k - 1).is_none(),
            "k not a multiple of QK_FP8"
        );
        assert!(
            gpu::gemv_operands(&f.e4m3[..f.e4m3.len() - 1], k, n_rows, n_rows, k).is_none(),
            "one block short"
        );
        assert!(
            gpu::gemv_operands(&f.e4m3, k - 1, n_rows, n_rows, k).is_none(),
            "input shorter than k"
        );
        assert!(
            gpu::gemv_operands(&f.e4m3, k, n_rows - 1, n_rows, k).is_none(),
            "output shorter than n_rows"
        );
        // The 4 GiB / 32-bit limits are not asserted here: no `blocks` slice
        // a test can build is long enough for them to be what declines the
        // call. The next test pins them on `gpu::weight_image`, which
        // `gemv_operands` consults before it looks at `blocks` at all.
    }

    /// `gpu::weight_image` — the sizing every FP8 GPU GEMV launch is cleared
    /// by — tested on the arithmetic itself, at shapes no test could allocate
    /// a weight matrix for. This is what pins the kernels' 32-bit limit:
    /// through `gemv_operands`, a 4 GiB shape is declined anyway for being
    /// longer than any `blocks` slice a test can build, limit or no limit.
    #[cfg(fp8_gpu_gemv)]
    #[test]
    fn gpu_weight_image_refuses_every_shape_the_kernels_cannot_address() {
        use gpu::{weight_image, WeightImage};
        use oxibonsai_core::BLOCK_FP8_BYTES;

        // Nothing to launch.
        assert_eq!(weight_image(0, QK_FP8), None, "no rows");
        assert_eq!(weight_image(1, 0), None, "k = 0");
        assert_eq!(
            weight_image(1, QK_FP8 + 1),
            None,
            "k not a multiple of QK_FP8"
        );

        // An ordinary shape: `n_rows · k / QK_FP8` blocks of 34 bytes.
        assert_eq!(
            weight_image(5, 3 * QK_FP8),
            Some(WeightImage {
                blocks: 15,
                bytes: 15 * BLOCK_FP8_BYTES
            })
        );

        // The largest image the kernels' 32-bit byte offsets reach — as tall
        // (one block per row) or as wide (one row) as it gets — is accepted
        // whole, and one block more takes it to 4 GiB or past it, which is
        // refused: `u32::MAX / 34 + 1` rows of `k = QK_FP8`, or one row that
        // many blocks wide.
        let max_blocks = u32::MAX as usize / BLOCK_FP8_BYTES;
        let largest = Some(WeightImage {
            blocks: max_blocks,
            bytes: max_blocks * BLOCK_FP8_BYTES,
        });
        assert_eq!(weight_image(max_blocks, QK_FP8), largest, "tallest image");
        assert_eq!(
            weight_image(1, max_blocks * QK_FP8),
            largest,
            "widest image"
        );
        assert_eq!(
            weight_image(max_blocks + 1, QK_FP8),
            None,
            "4 GiB weight image, one block per row"
        );
        assert_eq!(
            weight_image(1, (max_blocks + 1) * QK_FP8),
            None,
            "4 GiB weight image, one row"
        );

        // `n_rows` or `k` past `u32` (the kernels' parameter width) fails the
        // same byte bound — neither needs a check of its own (see
        // `weight_image`'s doc).
        #[cfg(target_pointer_width = "64")]
        {
            let past_u32 = u32::MAX as usize + 1;
            assert_eq!(weight_image(past_u32, QK_FP8), None, "n_rows past u32");
            assert_eq!(weight_image(1, past_u32), None, "k past u32");
        }

        // A block count or byte length that overflows `usize` is refused,
        // not wrapped into a small, plausible image: `2^w` blocks would wrap
        // to 0 blocks, and `(usize::MAX / 34 + 1) · 34` bytes to 16.
        assert_eq!(
            weight_image(usize::MAX / 2 + 1, 2 * QK_FP8),
            None,
            "block count overflows usize"
        );
        assert_eq!(
            weight_image(usize::MAX / BLOCK_FP8_BYTES + 1, QK_FP8),
            None,
            "byte length overflows usize"
        );
    }

    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    #[test]
    fn only_a_missing_cuda_device_is_a_silent_fallback() {
        use crate::gpu_backend::CudaGraphError;

        assert!(gpu::is_no_cuda_device(&CudaGraphError::DeviceNotFound(
            "device 0: CUDA_ERROR_NO_DEVICE".into()
        )));
        let faults = [
            CudaGraphError::CompilationFailed("nvrtc".into()),
            CudaGraphError::DriverError("gemv_fp8_e4m3 launch".into()),
            CudaGraphError::WeightNotFound(7),
            CudaGraphError::WeightLayoutError("FP8 E4M3 GEMV: k=31".into()),
            CudaGraphError::InvalidDimensions("k".into()),
            CudaGraphError::LockPoisoned,
            // The old test matched the `Display` text: a fault whose payload
            // merely mentions it must keep its warning.
            CudaGraphError::DriverError("no CUDA device: lookalike".into()),
        ];
        for fault in &faults {
            assert!(
                !gpu::is_no_cuda_device(fault),
                "`{fault}` is a fault and must keep its warning"
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn only_a_missing_metal_device_is_a_silent_fallback() {
        use crate::gpu_backend::MetalGraphError;

        assert!(gpu::is_no_metal_device(&MetalGraphError::DeviceNotFound));
        let faults = [
            MetalGraphError::CompilationFailed("msl".into()),
            MetalGraphError::BufferCreationFailed,
            MetalGraphError::EncodingFailed("k = 31 must be a non-zero multiple of 32".into()),
            MetalGraphError::ExecutionFailed("timeout".into()),
            MetalGraphError::InvalidDimensions("k".into()),
            // The old test matched the `Display` text: a fault whose payload
            // merely mentions it must keep its warning.
            MetalGraphError::ExecutionFailed("no Metal-capable GPU device found".into()),
        ];
        for fault in &faults {
            assert!(
                !gpu::is_no_metal_device(fault),
                "`{fault}` is a fault and must keep its warning"
            );
        }
    }
}
