//! scirs2-core GPU backend implementation.
//!
//! `Scirs2Backend` wraps a `scirs2_core::gpu::GpuContext` and implements
//! the [`GpuBackend`](super::GpuBackendTrait) trait, providing real GPU
//! acceleration via Metal (macOS) or CUDA kernels.
//!
//! Compiled kernels are cached on first use so subsequent calls avoid
//! recompilation.

#[cfg(feature = "gpu")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "gpu")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "gpu")]
use scirs2_core::gpu::{
    GpuBackend as ScirGpuBackend, GpuBuffer, GpuContext, GpuDataType, GpuError as ScirGpuError,
    GpuKernelHandle,
};

#[cfg(feature = "gpu")]
use tracing::{debug, info, warn};

#[cfg(feature = "gpu")]
use super::{DeviceBuffer, GpuError};

#[cfg(feature = "gpu")]
use super::GpuBackendTrait;

// ─────────────────────────────────────────────────────────────────────────
// Compiled kernel cache
// ─────────────────────────────────────────────────────────────────────────

/// Pre-compiled kernel handles (lazily initialised per context).
#[cfg(feature = "gpu")]
struct CompiledKernels {
    matvec_f32: GpuKernelHandle,
    relu: GpuKernelHandle,
    softmax: GpuKernelHandle,
    #[allow(dead_code)]
    rmsnorm: GpuKernelHandle,
    #[allow(dead_code)]
    silu: GpuKernelHandle,
}

/// Q1_0_g128-specific compiled kernels.
#[cfg(feature = "gpu")]
struct Q1Kernels {
    gemv: GpuKernelHandle,
    gemm: GpuKernelHandle,
}

/// TQ2_0_g128-specific (ternary) compiled kernels.
#[cfg(feature = "gpu")]
struct TQ2Kernels {
    gemv: GpuKernelHandle,
}

/// Helper kernels for LLM layer operations (SwiGLU, residual add, RMSNorm weighted).
#[cfg(feature = "gpu")]
struct HelperKernels {
    swiglu_fused: GpuKernelHandle,
    residual_add: GpuKernelHandle,
    rmsnorm_weighted: GpuKernelHandle,
}

/// Pre-allocated GPU buffers for the FFN pipeline phase.
/// Allocated lazily on first use and reused for all subsequent calls.
#[cfg(feature = "gpu")]
struct PipelineBuffers {
    hidden: GpuBuffer<f32>,
    attn_out: GpuBuffer<f32>,
    norm_weight: GpuBuffer<f32>,
    attn_proj: GpuBuffer<f32>,
    normed: GpuBuffer<f32>,
    gate_up: GpuBuffer<f32>,
    swiglu: GpuBuffer<f32>,
    down: GpuBuffer<f32>,
}

// ─────────────────────────────────────────────────────────────────────────
// Global singleton
// ─────────────────────────────────────────────────────────────────────────

/// Process-wide cached GPU backend.
///
/// Creating a [`GpuContext`] is expensive (Metal command queue creation +
/// kernel compilation), so we cache the backend globally and reuse it for
/// every GPU operation.
#[cfg(feature = "gpu")]
static GLOBAL_BACKEND: OnceLock<Mutex<Option<Arc<Scirs2Backend>>>> = OnceLock::new();

/// Monotonically increasing handle ID generator for weight cache entries.
#[cfg(feature = "gpu")]
static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

// ─────────────────────────────────────────────────────────────────────────
// Scirs2Backend
// ─────────────────────────────────────────────────────────────────────────

/// GPU-accelerated backend via scirs2-core.
///
/// Wraps a [`GpuContext`] and compiles GPU kernels on first use.
/// Falls back to CPU on kernel compilation failure.
///
/// # Feature gates
///
/// - `gpu`   – generic GPU support (CPU fallback in scirs2-core)
/// - `metal` – Apple Metal backend (macOS only)
/// - `cuda`  – NVIDIA CUDA backend
#[cfg(feature = "gpu")]
pub struct Scirs2Backend {
    ctx: GpuContext,
    /// Lazily compiled general-purpose kernels.
    kernels: OnceLock<Result<CompiledKernels, String>>,
    /// Lazily compiled Q1_0_g128 kernels.
    q1_kernels: OnceLock<Result<Q1Kernels, String>>,
    /// Lazily compiled TQ2_0_g128 (ternary) kernels.
    tq2_kernels: OnceLock<Result<TQ2Kernels, String>>,
    /// Cached GPU-resident weight buffers, keyed by [`GpuWeightHandle`] ID,
    /// deduplicated by content and reference-counted per model epoch (see
    /// [`WeightCacheState`]).
    weight_cache: Mutex<WeightCacheState>,
    /// Whether [`upload_weights`](Self::upload_weights) /
    /// [`upload_weights_ternary`](Self::upload_weights_ternary) retain their
    /// uploaded buffer in `weight_cache` for later
    /// [`gemv_q1_g128_cached`](Self::gemv_q1_g128_cached) /
    /// [`gemv_tq2_g128_cached`](Self::gemv_tq2_g128_cached) lookups.
    ///
    /// Defaults to `true` (today's behaviour, unchanged). MET-M1: this cache
    /// is a **second**, independently-keyed, full GPU-resident copy of every
    /// quantized tensor — the fused `MetalGraph` decode path
    /// (`oxibonsai-model`) maintains its own weight cache (see MET-02), and
    /// when that fused path is active for a model, `Scirs2Backend`'s copy is
    /// redundant (measured 435.69 MB on the 1.7B, uploaded unconditionally
    /// at engine construction). Set to `false` via
    /// [`with_weight_cache_enabled`](Self::with_weight_cache_enabled) or
    /// [`set_weight_cache_enabled`](Self::set_weight_cache_enabled) so the
    /// upload site can keep calling `upload_weights{,_ternary}`
    /// unconditionally (e.g. for its other side effects) without retaining a
    /// second GPU-resident copy.
    weight_cache_enabled: std::sync::atomic::AtomicBool,
    /// Serialisation point for every `upload → bind → dispatch → read back`
    /// critical section, and owner of the reusable I/O buffers it protects.
    ///
    /// Both the pooled buffers and the compiled [`GpuKernelHandle`]s are
    /// process-global, and the handles take `&self` for `set_buffer` /
    /// `set_u32`, so their parameter bindings are shared mutable state too;
    /// the whole sequence must be atomic against other callers. It was not:
    /// `get_input_buf` / `get_output_buf` released their mutex *before*
    /// returning the buffer, so two concurrent `gemv_q1_g128` /
    /// `gemm_q1_g128` calls on [`Scirs2Backend::global()`](Self::global)
    /// overwrote each other's input, kernel parameters and output —
    /// non-deterministic wrong numbers under `cargo test`'s in-process
    /// parallelism, invisible to `cargo nextest` (one process per test).
    ///
    /// **Trade-off:** GPU dispatch through `Scirs2Backend` is strictly
    /// serial. The shipping fused decode paths do not go through here — since
    /// `MET-08` they run in per-replica `MetalGraph` sessions, each with its
    /// own command queue — so what this mutex serialises is the per-layer
    /// block-dispatch fallback and the Q1 batch phases, every one of which
    /// ends in a synchronous readback of a shared I/O buffer. Per-call I/O
    /// buffers would permit concurrency there but reintroduce the allocation
    /// churn the pool exists to remove *and* leave the kernel-parameter race
    /// untouched. Lock order: `dispatch` is acquired **last**, after any
    /// `weight_cache` / `pipeline_buffers` guard is dropped.
    dispatch: Mutex<DispatchState>,
    /// Lazily compiled helper kernels (SwiGLU, residual_add, RMSNorm weighted).
    helper_kernels: OnceLock<Result<HelperKernels, String>>,
    /// Pre-allocated pipeline buffers for FFN dispatch operations.
    pipeline_buffers: Mutex<Option<PipelineBuffers>>,
}

// ─────────────────────────────────────────────────────────────────────────
// Content-deduplicated, epoch-refcounted weight cache (MET-M1 / Q1 replicas)
// ─────────────────────────────────────────────────────────────────────────

// The cache state, its content fingerprint and the MetalGraph mirror
// eviction live in a sibling file so this module stays under the 2000-line
// ceiling; `#[path]` keeps it a child of `scirs2_backend` (same pattern as
// the test module below).
#[cfg(feature = "gpu")]
#[path = "scirs2_weight_cache.rs"]
mod weight_cache_state;
#[cfg(all(feature = "metal", target_os = "macos"))]
use weight_cache_state::release_metal_mirrors;
#[cfg(feature = "gpu")]
use weight_cache_state::{
    content_fingerprint, resident_bytes_equal, ContentKey, UploadLayout, WeightCacheState,
};

/// Shared dispatch state: the reusable I/O buffers, behind the mutex that
/// also serialises kernel binding and dispatch. Capacities live here rather
/// than in atomics so "is it big enough?" and "replace it" are one
/// indivisible decision.
#[cfg(feature = "gpu")]
#[derive(Default)]
struct DispatchState {
    /// Reusable device-side input buffer, or `None` before first use.
    input: Option<GpuBuffer<f32>>,
    /// Element capacity of `input`.
    input_capacity: usize,
    /// Reusable device-side output buffer, or `None` before first use.
    output: Option<GpuBuffer<f32>>,
    /// Element capacity of `output`.
    output_capacity: usize,
}

#[cfg(feature = "gpu")]
impl DispatchState {
    /// Reusable input buffer holding at least `required` f32 elements,
    /// growing (never shrinking) when it is too small.
    fn input_buf(&mut self, ctx: &GpuContext, required: usize) -> GpuBuffer<f32> {
        if self.input_capacity >= required {
            if let Some(buf) = self.input.as_ref() {
                return buf.clone();
            }
        }
        let new_cap = Self::grown(self.input_capacity, required);
        let buf = ctx.create_buffer::<f32>(new_cap);
        self.input_capacity = new_cap;
        self.input = Some(buf.clone());
        buf
    }

    /// Reusable output buffer holding at least `required` f32 elements.
    fn output_buf(&mut self, ctx: &GpuContext, required: usize) -> GpuBuffer<f32> {
        if self.output_capacity >= required {
            if let Some(buf) = self.output.as_ref() {
                return buf.clone();
            }
        }
        let new_cap = Self::grown(self.output_capacity, required);
        let buf = ctx.create_buffer::<f32>(new_cap);
        self.output_capacity = new_cap;
        self.output = Some(buf.clone());
        buf
    }

    /// Growth policy, unchanged from the pre-fix pool: at least 16 K
    /// elements, otherwise `max(required, 2 × current)` so a ramping shape
    /// reallocates O(log n) times rather than per call.
    fn grown(current: usize, required: usize) -> usize {
        required.max(current.saturating_mul(2)).max(16384)
    }
}

#[cfg(feature = "gpu")]
impl Scirs2Backend {
    /// Attempt to create a scirs2-core GPU context.
    ///
    /// Tries Metal on macOS (if `metal` feature), then CUDA (if `cuda`
    /// feature), then the scirs2-core CPU backend.
    pub fn new() -> Result<Self, GpuError> {
        let backend = Self::detect_backend();
        let ctx = GpuContext::new(backend).map_err(|e| Self::convert_error(&e))?;
        info!(
            "Scirs2Backend initialised with {} backend",
            ctx.backend_name()
        );
        Ok(Self {
            ctx,
            kernels: OnceLock::new(),
            q1_kernels: OnceLock::new(),
            tq2_kernels: OnceLock::new(),
            weight_cache: Mutex::new(WeightCacheState::default()),
            weight_cache_enabled: std::sync::atomic::AtomicBool::new(true),
            dispatch: Mutex::new(DispatchState::default()),
            helper_kernels: OnceLock::new(),
            pipeline_buffers: Mutex::new(None),
        })
    }

    /// Get or create the global GPU backend singleton.
    ///
    /// The first call creates a [`GpuContext`] and compiles kernels lazily.
    /// Subsequent calls return an `Arc` clone of the same backend, avoiding
    /// the overhead of re-initialising Metal/CUDA contexts.
    pub fn global() -> Result<Arc<Self>, GpuError> {
        let mutex = GLOBAL_BACKEND.get_or_init(|| Mutex::new(None));
        let mut guard = mutex
            .lock()
            .map_err(|_| GpuError::NotAvailable("GPU backend lock poisoned".into()))?;
        if let Some(ref cached) = *guard {
            return Ok(Arc::clone(cached));
        }
        let backend = Arc::new(Self::new()?);
        *guard = Some(Arc::clone(&backend));
        Ok(backend)
    }

    /// Create with a specific scirs2-core backend.
    pub fn with_backend(backend: ScirGpuBackend) -> Result<Self, GpuError> {
        let ctx = GpuContext::new(backend).map_err(|e| Self::convert_error(&e))?;
        info!(
            "Scirs2Backend initialised with {} backend (explicit)",
            ctx.backend_name()
        );
        Ok(Self {
            ctx,
            kernels: OnceLock::new(),
            q1_kernels: OnceLock::new(),
            tq2_kernels: OnceLock::new(),
            weight_cache: Mutex::new(WeightCacheState::default()),
            weight_cache_enabled: std::sync::atomic::AtomicBool::new(true),
            dispatch: Mutex::new(DispatchState::default()),
            helper_kernels: OnceLock::new(),
            pipeline_buffers: Mutex::new(None),
        })
    }

    /// Returns the scirs2-core backend name.
    pub fn backend_name(&self) -> &str {
        self.ctx.backend_name()
    }

    // ── Weight-cache opt-in (MET-M1) ─────────────────────────────────────

    /// Builder-style: set whether [`upload_weights`](Self::upload_weights) /
    /// [`upload_weights_ternary`](Self::upload_weights_ternary) retain their
    /// uploaded buffer for later cached-GEMV lookups (see
    /// `weight_cache_enabled`'s field docs). Composes with [`new`](Self::new)
    /// / [`with_backend`](Self::with_backend):
    ///
    /// ```ignore
    /// let backend = Scirs2Backend::new()?.with_weight_cache_enabled(false);
    /// ```
    #[must_use]
    pub fn with_weight_cache_enabled(self, enabled: bool) -> Self {
        self.weight_cache_enabled.store(enabled, Ordering::Relaxed);
        self
    }

    /// Enable or disable weight-cache retention on an already-constructed
    /// (possibly shared, e.g. the [`global`](Self::global) singleton)
    /// backend. See `weight_cache_enabled`'s field docs.
    pub fn set_weight_cache_enabled(&self, enabled: bool) {
        self.weight_cache_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Whether weight-cache retention is currently enabled.
    pub fn weight_cache_enabled(&self) -> bool {
        self.weight_cache_enabled.load(Ordering::Relaxed)
    }

    // ── Backend detection ────────────────────────────────────────────────

    fn detect_backend() -> ScirGpuBackend {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if ScirGpuBackend::Metal.is_available() {
                debug!("Metal backend detected");
                return ScirGpuBackend::Metal;
            }
        }
        #[cfg(feature = "cuda")]
        {
            if ScirGpuBackend::Cuda.is_available() {
                debug!("CUDA backend detected");
                return ScirGpuBackend::Cuda;
            }
        }
        debug!("Falling back to scirs2-core CPU backend");
        ScirGpuBackend::Cpu
    }

    // ── Kernel compilation (lazy) ───────────────────────────────────────

    /// Get or compile the general-purpose kernels (matvec, relu, softmax, …).
    fn get_kernels(&self) -> Result<&CompiledKernels, GpuError> {
        let result = self.kernels.get_or_init(|| self.compile_base_kernels());
        match result {
            Ok(k) => Ok(k),
            Err(msg) => Err(GpuError::KernelLaunch(msg.clone())),
        }
    }

    /// Get or compile the Q1_0_g128 kernels (gemv, gemm).
    fn get_q1_kernels(&self) -> Result<&Q1Kernels, GpuError> {
        let result = self.q1_kernels.get_or_init(|| self.compile_q1_kernels());
        match result {
            Ok(k) => Ok(k),
            Err(msg) => Err(GpuError::KernelLaunch(msg.clone())),
        }
    }

    fn compile_base_kernels(&self) -> Result<CompiledKernels, String> {
        let (matvec_src, relu_src, softmax_src, rmsnorm_src, silu_src) =
            Self::select_base_sources();

        let compile = |name: &str, src: &str| -> Result<GpuKernelHandle, String> {
            self.ctx
                .execute(|compiler| compiler.compile(src))
                .map_err(|e| format!("{name}: {e}"))
        };

        Ok(CompiledKernels {
            matvec_f32: compile("matvec_f32", matvec_src)?,
            relu: compile("relu", relu_src)?,
            softmax: compile("softmax", softmax_src)?,
            rmsnorm: compile("rmsnorm", rmsnorm_src)?,
            silu: compile("silu", silu_src)?,
        })
    }

    fn compile_q1_kernels(&self) -> Result<Q1Kernels, String> {
        let (gemv_src, gemm_src) = Self::select_q1_sources();

        let compile = |name: &str, src: &str| -> Result<GpuKernelHandle, String> {
            self.ctx
                .execute(|compiler| compiler.compile(src))
                .map_err(|e| format!("{name}: {e}"))
        };

        Ok(Q1Kernels {
            gemv: compile("gemv_q1_g128", gemv_src)?,
            gemm: compile("gemm_q1_g128", gemm_src)?,
        })
    }

    /// Get or compile the TQ2_0_g128 (ternary) kernels.
    fn get_tq2_kernels(&self) -> Result<&TQ2Kernels, GpuError> {
        let result = self.tq2_kernels.get_or_init(|| self.compile_tq2_kernels());
        match result {
            Ok(k) => Ok(k),
            Err(msg) => Err(GpuError::KernelLaunch(msg.clone())),
        }
    }

    fn compile_tq2_kernels(&self) -> Result<TQ2Kernels, String> {
        let gemv_src = Self::select_tq2_gemv_source();
        if gemv_src.is_empty() {
            return Err("TQ2_0_g128 GPU GEMV is only implemented for Metal (macOS); no CUDA ternary kernel available".into());
        }
        let compile = |name: &str, src: &str| -> Result<GpuKernelHandle, String> {
            self.ctx
                .execute(|compiler| compiler.compile(src))
                .map_err(|e| format!("{name}: {e}"))
        };
        Ok(TQ2Kernels {
            gemv: compile("gemv_tq2_g128_v1", gemv_src)?,
        })
    }

    /// Returns the TQ2_0_g128 GEMV kernel source for the active backend.
    #[allow(unreachable_code)]
    fn select_tq2_gemv_source() -> &'static str {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            use super::kernel_sources::MSL_GEMV_TQ2_G128_V1;
            return MSL_GEMV_TQ2_G128_V1;
        }
        // No CUDA ternary kernel yet — return empty string (compile will fail gracefully).
        ""
    }

    fn get_helper_kernels(&self) -> Result<&HelperKernels, GpuError> {
        let result = self
            .helper_kernels
            .get_or_init(|| self.compile_helper_kernels());
        result
            .as_ref()
            .map_err(|msg| GpuError::NotAvailable(format!("helper kernel compile: {msg}")))
    }

    #[allow(unreachable_code)]
    fn compile_helper_kernels(&self) -> Result<HelperKernels, String> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            use super::kernel_sources::*;
            let compile = |name: &str, src: &str| -> Result<GpuKernelHandle, String> {
                self.ctx
                    .execute(|compiler| compiler.compile(src))
                    .map_err(|e| format!("{name}: {e}"))
            };
            return Ok(HelperKernels {
                swiglu_fused: compile("swiglu_fused", MSL_SWIGLU_FUSED)?,
                residual_add: compile("residual_add", MSL_RESIDUAL_ADD)?,
                rmsnorm_weighted: compile("rmsnorm_weighted", MSL_RMSNORM_WEIGHTED)?,
            });
        }
        #[cfg(feature = "cuda")]
        {
            use super::kernel_sources::*;
            let compile = |name: &str, src: &str| -> Result<GpuKernelHandle, String> {
                self.ctx
                    .execute(|compiler| compiler.compile(src))
                    .map_err(|e| format!("{name}: {e}"))
            };
            return Ok(HelperKernels {
                swiglu_fused: compile("swiglu_fused", CUDA_SWIGLU_FUSED)?,
                residual_add: compile("residual_add", CUDA_RESIDUAL_ADD)?,
                rmsnorm_weighted: compile("rmsnorm_weighted", CUDA_RMSNORM_WEIGHTED)?,
            });
        }
        #[allow(unreachable_code)]
        Err("no GPU backend available for helper kernels".into())
    }

    // ── Source selection per backend ─────────────────────────────────────

    /// Returns (matvec, relu, softmax, rmsnorm, silu) kernel sources.
    #[allow(unreachable_code)]
    fn select_base_sources() -> (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    ) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            use super::kernel_sources::*;
            return (MSL_MATVEC_F32, MSL_RELU, MSL_SOFTMAX, MSL_RMSNORM, MSL_SILU);
        }
        #[cfg(feature = "cuda")]
        {
            use super::kernel_sources::*;
            return (
                CUDA_MATVEC_F32,
                CUDA_RELU,
                CUDA_SOFTMAX,
                CUDA_RMSNORM,
                CUDA_SILU,
            );
        }
        #[allow(unreachable_code)]
        {
            // CPU fallback: empty sources (should not be compiled).
            ("", "", "", "", "")
        }
    }

    /// Returns (gemv, gemm) Q1_0_g128 kernel sources.
    #[allow(unreachable_code)]
    fn select_q1_sources() -> (&'static str, &'static str) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            use super::kernel_sources::*;
            return (MSL_GEMV_Q1_G128, MSL_GEMM_Q1_G128);
        }
        #[cfg(feature = "cuda")]
        {
            use super::kernel_sources::*;
            return (CUDA_GEMV_Q1_G128, CUDA_GEMM_Q1_G128);
        }
        #[allow(unreachable_code)]
        {
            ("", "")
        }
    }

    // ── Error conversion ────────────────────────────────────────────────

    fn convert_error(e: &ScirGpuError) -> GpuError {
        match e {
            ScirGpuError::BackendNotAvailable(msg) => GpuError::NotAvailable(msg.clone()),
            ScirGpuError::OutOfMemory(_msg) => GpuError::OutOfMemory {
                requested: 0,
                device: 0,
            },
            ScirGpuError::KernelCompilationError(msg) => {
                GpuError::KernelLaunch(format!("compilation: {msg}"))
            }
            ScirGpuError::KernelExecutionError(msg) => {
                GpuError::KernelLaunch(format!("execution: {msg}"))
            }
            ScirGpuError::InvalidParameter(msg) => GpuError::InvalidArgument(msg.clone()),
            _ => GpuError::NotAvailable(format!("{e}")),
        }
    }

    // ── Helper: workgroup count ─────────────────────────────────────────

    /// Number of threadgroups needed to cover `n` elements with 256 threads.
    fn workgroups_1d(n: usize) -> u32 {
        let n = n as u32;
        (n.saturating_add(255)) / 256
    }

    /// For simdgroup-based GEMV kernel: ceil(n_rows / 8) workgroups.
    fn workgroups_simd(n_rows: usize) -> u32 {
        (n_rows as u32).div_ceil(8)
    }

    // ── Dispatch serialisation + reusable I/O buffers ───────────────────

    /// Acquire the dispatch lock; hold the guard across the **entire**
    /// upload → bind → dispatch → readback sequence, and acquire it last,
    /// after any `weight_cache` / `pipeline_buffers` guard is dropped (see
    /// [`Scirs2Backend::dispatch`]).
    fn lock_dispatch(&self) -> Result<std::sync::MutexGuard<'_, DispatchState>, GpuError> {
        self.dispatch
            .lock()
            .map_err(|_| GpuError::NotAvailable("GPU dispatch lock poisoned".into()))
    }

    /// Read back exactly `count` f32 elements from a (possibly oversized) GPU buffer.
    fn read_output(buf: &GpuBuffer<f32>, count: usize) -> Result<Vec<f32>, GpuError> {
        let mut result = vec![0.0_f32; count];
        buf.copy_to_host(&mut result)
            .map_err(|e| GpuError::KernelLaunch(format!("output readback: {e}")))?;
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// GpuBackendTrait implementation
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
impl GpuBackendTrait for Scirs2Backend {
    fn name(&self) -> &'static str {
        // `F7` (CUDA-SAFETY residue): this used to be a `cfg!` ladder, so a
        // build with the `cuda` feature reported "scirs2-cuda" even when the
        // context had actually fallen back to the CPU backend — a name that
        // then appeared in logs, `/admin/status` and tier-selection warnings.
        // Ask the context what it really is. The trait wants a `&'static str`,
        // so map the backend enum rather than borrowing `backend_name()`.
        match self.ctx.backend() {
            ScirGpuBackend::Metal => "scirs2-metal",
            ScirGpuBackend::Cuda => "scirs2-cuda",
            ScirGpuBackend::Rocm => "scirs2-rocm",
            ScirGpuBackend::Wgpu => "scirs2-wgpu",
            ScirGpuBackend::OpenCL => "scirs2-opencl",
            ScirGpuBackend::Cpu => "scirs2-cpu",
        }
    }

    fn is_accelerated(&self) -> bool {
        // Only Metal (macOS) has a fully-implemented kernel-launch path in
        // scirs2-core.  The CUDA backend's `execute_cuda_kernel()` sets up
        // parameters and calculates launch config but never calls
        // `func.launch()` — it is an incomplete stub that leaves output
        // buffers unmodified, producing garbage tokens.
        //
        // Returning `false` here for non-Metal backends causes
        // `KernelDispatcher::auto_detect()` to fall through to the CPU SIMD
        // tier (AVX-512 / AVX2 on x86-64), which produces correct results.
        // Once the CUDA kernel-launch path is complete in scirs2-core, remove
        // the cfg gate below so CUDA is also treated as accelerated.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if matches!(self.ctx.backend(), ScirGpuBackend::Metal) {
                return true;
            }
        }
        // CUDA or scirs2-core CPU context: not hardware-accelerated via this path.
        if !matches!(self.ctx.backend(), ScirGpuBackend::Cpu) {
            warn!(
                backend = self.ctx.backend_name(),
                "GPU context initialised but kernel dispatch is not yet \
                 implemented for this backend — falling back to CPU SIMD"
            );
        }
        false
    }

    fn device_count(&self) -> usize {
        1 // scirs2-core currently exposes one device per context
    }

    fn alloc(&self, size: usize, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        Ok(DeviceBuffer::new(size, device_id))
    }

    fn host_to_device(&self, src: &[f32], device_id: usize) -> Result<DeviceBuffer, GpuError> {
        Ok(DeviceBuffer::from_slice(src, device_id))
    }

    fn device_to_host(&self, buf: &DeviceBuffer) -> Result<Vec<f32>, GpuError> {
        Ok(buf.to_vec())
    }

    fn matvec(
        &self,
        a: &DeviceBuffer,
        x: &DeviceBuffer,
        m: usize,
        k: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        if a.size() != m * k {
            return Err(GpuError::InvalidArgument(format!(
                "matrix buffer size {} does not match m={} k={}",
                a.size(),
                m,
                k
            )));
        }
        if x.size() != k {
            return Err(GpuError::InvalidArgument(format!(
                "vector buffer size {} does not match k={}",
                x.size(),
                k
            )));
        }

        // Try GPU kernel path first.
        match self.matvec_gpu(a, x, m, k) {
            Ok(result) => return Ok(DeviceBuffer::from_slice(&result, device_id)),
            Err(e) => {
                warn!("Scirs2Backend matvec GPU path failed ({e}), falling back to CPU");
            }
        }

        // CPU fallback.
        Self::matvec_cpu(a, x, m, k, device_id)
    }

    fn relu(&self, x: &DeviceBuffer, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        match self.relu_gpu(x) {
            Ok(result) => return Ok(DeviceBuffer::from_slice(&result, device_id)),
            Err(e) => {
                warn!("Scirs2Backend relu GPU path failed ({e}), falling back to CPU");
            }
        }
        Self::relu_cpu(x, device_id)
    }

    fn softmax(
        &self,
        x: &DeviceBuffer,
        size: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        if x.size() != size {
            return Err(GpuError::InvalidArgument(format!(
                "buffer size {} does not match size={}",
                x.size(),
                size
            )));
        }
        if size == 0 {
            return Ok(DeviceBuffer::new(0, device_id));
        }

        match self.softmax_gpu(x, size) {
            Ok(result) => return Ok(DeviceBuffer::from_slice(&result, device_id)),
            Err(e) => {
                warn!("Scirs2Backend softmax GPU path failed ({e}), falling back to CPU");
            }
        }
        Self::softmax_cpu(x, size, device_id)
    }

    fn synchronize(&self, _device_id: usize) -> Result<(), GpuError> {
        // scirs2-core's dispatch is synchronous (wait_until_completed).
        Ok(())
    }

    fn memory_info(&self, _device_id: usize) -> Result<(usize, usize), GpuError> {
        let total = self
            .ctx
            .get_total_memory()
            .unwrap_or(4 * 1024 * 1024 * 1024);
        let free = self.ctx.get_available_memory().unwrap_or(total / 2);
        Ok((free, total))
    }

    fn release_model(&self, model_epoch: u64) -> Result<usize, GpuError> {
        self.release_model_weights(model_epoch)
    }

    fn resident_weight_bytes(&self) -> u64 {
        self.cached_weight_bytes()
    }

    fn resident_weight_count(&self) -> usize {
        self.cached_weight_count()
    }

    fn weight_bytes_uploaded_total(&self) -> u64 {
        self.uploaded_weight_bytes_total()
    }

    fn model_registration_count(&self, model_epoch: u64) -> usize {
        self.epoch_registration_count(model_epoch)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// GPU kernel dispatch helpers
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
impl Scirs2Backend {
    /// GPU-accelerated FP32 matrix-vector multiply.
    fn matvec_gpu(
        &self,
        a: &DeviceBuffer,
        x: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let kernels = self.get_kernels()?;

        let a_buf: GpuBuffer<f32> = self.ctx.create_buffer_from_slice(&a.data);
        let x_buf: GpuBuffer<f32> = self.ctx.create_buffer_from_slice(&x.data);
        let out_buf: GpuBuffer<f32> = self.ctx.create_buffer::<f32>(m);

        // Bind → dispatch → readback is one critical section: `kernels` is
        // process-global and its parameter bindings are shared mutable state.
        let _dispatch = self.lock_dispatch()?;
        kernels.matvec_f32.set_buffer("x", &a_buf);
        kernels.matvec_f32.set_buffer("y", &x_buf);
        kernels.matvec_f32.set_buffer("result", &out_buf);
        kernels.matvec_f32.set_u32("n", m as u32);
        kernels.matvec_f32.set_u32("k", k as u32);

        kernels.matvec_f32.dispatch([Self::workgroups_1d(m), 1, 1]);

        Ok(out_buf.to_vec())
    }

    /// GPU-accelerated ReLU.
    fn relu_gpu(&self, x: &DeviceBuffer) -> Result<Vec<f32>, GpuError> {
        let kernels = self.get_kernels()?;
        let n = x.size();

        let in_buf: GpuBuffer<f32> = self.ctx.create_buffer_from_slice(&x.data);
        let out_buf: GpuBuffer<f32> = self.ctx.create_buffer::<f32>(n);

        let _dispatch = self.lock_dispatch()?;
        kernels.relu.set_buffer("x", &in_buf);
        kernels.relu.set_buffer("result", &out_buf);
        kernels.relu.set_u32("n", n as u32);

        kernels.relu.dispatch([Self::workgroups_1d(n), 1, 1]);

        Ok(out_buf.to_vec())
    }

    /// GPU-accelerated softmax.
    fn softmax_gpu(&self, x: &DeviceBuffer, size: usize) -> Result<Vec<f32>, GpuError> {
        let kernels = self.get_kernels()?;

        let in_buf: GpuBuffer<f32> = self.ctx.create_buffer_from_slice(&x.data);
        let out_buf: GpuBuffer<f32> = self.ctx.create_buffer::<f32>(size);

        let _dispatch = self.lock_dispatch()?;
        kernels.softmax.set_buffer("x", &in_buf);
        kernels.softmax.set_buffer("result", &out_buf);
        kernels.softmax.set_u32("n", size as u32);

        kernels.softmax.dispatch([Self::workgroups_1d(size), 1, 1]);

        Ok(out_buf.to_vec())
    }

    // ── CPU fallback implementations ────────────────────────────────────

    fn matvec_cpu(
        a: &DeviceBuffer,
        x: &DeviceBuffer,
        m: usize,
        k: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        let mut result = vec![0.0_f32; m];
        for (row, result_val) in result.iter_mut().enumerate().take(m) {
            let mut acc = 0.0_f32;
            for col in 0..k {
                acc += a.data[row * k + col] * x.data[col];
            }
            *result_val = acc;
        }
        Ok(DeviceBuffer::from_slice(&result, device_id))
    }

    fn relu_cpu(x: &DeviceBuffer, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        let result: Vec<f32> = x.data.iter().map(|&v| v.max(0.0)).collect();
        Ok(DeviceBuffer::from_slice(&result, device_id))
    }

    fn softmax_cpu(
        x: &DeviceBuffer,
        size: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        let max_val = x.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = x.data.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let result: Vec<f32> = if sum == 0.0 {
            vec![1.0 / size as f32; size]
        } else {
            exps.iter().map(|&e| e / sum).collect()
        };
        Ok(DeviceBuffer::from_slice(&result, device_id))
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Q1_0_g128 GPU operations
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
impl Scirs2Backend {
    /// Q1_0_g128 matrix-vector product on GPU.
    ///
    /// Dequantises 1-bit weights and multiplies by the input vector entirely
    /// on the GPU.
    ///
    /// # Arguments
    /// - `block_bytes` — raw bytes of `BlockQ1_0G128[]` (18 bytes per block)
    /// - `input` — FP32 input vector of length `k`
    /// - `n_rows` — number of output rows
    /// - `k` — input dimension (must be a multiple of 128)
    ///
    /// # Returns
    /// FP32 output vector of length `n_rows`.
    pub fn gemv_q1_g128(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        if k == 0 || !k.is_multiple_of(128) {
            return Err(GpuError::InvalidArgument(format!(
                "k={k} must be a positive multiple of 128"
            )));
        }
        let blocks_per_row = k / 128;
        let expected_bytes = n_rows * blocks_per_row * 18;
        if block_bytes.len() < expected_bytes {
            return Err(GpuError::InvalidArgument(format!(
                "block_bytes.len()={} < expected {} (n_rows={}, blocks_per_row={})",
                block_bytes.len(),
                expected_bytes,
                n_rows,
                blocks_per_row,
            )));
        }
        if input.len() != k {
            return Err(GpuError::InvalidArgument(format!(
                "input.len()={} != k={}",
                input.len(),
                k
            )));
        }

        let kernels = self.get_q1_kernels()?;

        let blocks_buf: GpuBuffer<u8> = self.ctx.create_buffer_from_slice(block_bytes);

        // Reuse the persistent I/O buffers, with the dispatch lock held
        // across upload → bind → dispatch → readback: both the buffers and
        // `kernels.gemv`'s parameter bindings are process-global, so a
        // concurrent caller that got in between any two of these steps used
        // to silently swap inputs or outputs with this one.
        let mut dispatch = self.lock_dispatch()?;
        let input_buf = dispatch.input_buf(&self.ctx, input.len());
        input_buf
            .copy_from_host(input)
            .map_err(|e| Self::convert_error(&e))?;
        let output_buf = dispatch.output_buf(&self.ctx, n_rows);

        kernels.gemv.set_buffer("x", &blocks_buf);
        kernels.gemv.set_buffer("y", &input_buf);
        kernels.gemv.set_buffer("result", &output_buf);
        kernels.gemv.set_u32("n", n_rows as u32);
        kernels.gemv.set_u32("k", k as u32);

        kernels.gemv.dispatch([Self::workgroups_1d(n_rows), 1, 1]);

        Self::read_output(&output_buf, n_rows)
    }

    /// Q1_0_g128 matrix-matrix product on GPU (batched GEMV).
    ///
    /// # Arguments
    /// - `block_bytes` — raw bytes of `BlockQ1_0G128[]`
    /// - `input` — FP32 input matrix of shape `[m, k]` (row-major)
    /// - `m` — batch size (number of input vectors)
    /// - `n_rows` — number of weight rows
    /// - `k` — inner dimension (multiple of 128)
    ///
    /// # Returns
    /// FP32 output matrix of shape `[m, n_rows]` (row-major).
    pub fn gemm_q1_g128(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        if k == 0 || !k.is_multiple_of(128) {
            return Err(GpuError::InvalidArgument(format!(
                "k={k} must be a positive multiple of 128"
            )));
        }
        let blocks_per_row = k / 128;
        let expected_bytes = n_rows * blocks_per_row * 18;
        if block_bytes.len() < expected_bytes {
            return Err(GpuError::InvalidArgument(format!(
                "block_bytes.len()={} < expected {}",
                block_bytes.len(),
                expected_bytes,
            )));
        }
        if input.len() != m * k {
            return Err(GpuError::InvalidArgument(format!(
                "input.len()={} != m*k={}",
                input.len(),
                m * k
            )));
        }

        let kernels = self.get_q1_kernels()?;

        let blocks_buf: GpuBuffer<u8> = self.ctx.create_buffer_from_slice(block_bytes);

        // Same critical section as `gemv_q1_g128`, and the *same* pooled
        // buffers and kernel object — the two entry points must serialise
        // against each other, not only against themselves.
        let out_elems = m * n_rows;
        let mut dispatch = self.lock_dispatch()?;
        let input_buf = dispatch.input_buf(&self.ctx, input.len());
        input_buf
            .copy_from_host(input)
            .map_err(|e| Self::convert_error(&e))?;
        let output_buf = dispatch.output_buf(&self.ctx, out_elems);

        kernels.gemm.set_buffer("x", &blocks_buf);
        kernels.gemm.set_buffer("y", &input_buf);
        kernels.gemm.set_buffer("result", &output_buf);
        kernels.gemm.set_u32("n", n_rows as u32);
        kernels.gemm.set_u32("m", m as u32);
        kernels.gemm.set_u32("k", k as u32);

        let wg_x = Self::workgroups_1d(n_rows);
        let wg_y = Self::workgroups_1d(m);
        kernels.gemm.dispatch([wg_x, wg_y, 1]);

        Self::read_output(&output_buf, out_elems)
    }

    /// RMSNorm on GPU.
    ///
    /// Computes `y[i] = x[i] / sqrt(mean(x²) + eps) * weight[i]`.
    pub fn rmsnorm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, GpuError> {
        let n = input.len();
        if weight.len() != n {
            return Err(GpuError::InvalidArgument(format!(
                "weight.len()={} != input.len()={}",
                weight.len(),
                n,
            )));
        }

        let kernels = self.get_kernels()?;

        let in_buf: GpuBuffer<f32> = self.ctx.create_buffer_from_slice(input);
        let w_buf: GpuBuffer<f32> = self.ctx.create_buffer_from_slice(weight);
        let out_buf: GpuBuffer<f32> = self.ctx.create_buffer::<f32>(n);

        let _dispatch = self.lock_dispatch()?;
        kernels.rmsnorm.set_buffer("x", &in_buf);
        kernels.rmsnorm.set_buffer("y", &w_buf);
        kernels.rmsnorm.set_buffer("result", &out_buf);
        kernels.rmsnorm.set_f32("alpha", eps);
        kernels.rmsnorm.set_u32("n", n as u32);

        kernels.rmsnorm.dispatch([Self::workgroups_1d(n), 1, 1]);

        Ok(out_buf.to_vec())
    }

    /// SiLU activation on GPU.
    ///
    /// Computes `y[i] = x[i] / (1 + exp(-x[i]))`.
    pub fn silu(&self, input: &[f32]) -> Result<Vec<f32>, GpuError> {
        let n = input.len();
        let kernels = self.get_kernels()?;

        let in_buf: GpuBuffer<f32> = self.ctx.create_buffer_from_slice(input);
        let out_buf: GpuBuffer<f32> = self.ctx.create_buffer::<f32>(n);

        let _dispatch = self.lock_dispatch()?;
        kernels.silu.set_buffer("x", &in_buf);
        kernels.silu.set_buffer("result", &out_buf);
        kernels.silu.set_u32("n", n as u32);

        kernels.silu.dispatch([Self::workgroups_1d(n), 1, 1]);

        Ok(out_buf.to_vec())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// GPU weight cache operations
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
impl Scirs2Backend {
    /// Upload weight block bytes to a GPU buffer and return a reusable handle.
    ///
    /// The returned [`GpuWeightHandle`](crate::weight_cache::GpuWeightHandle) can be passed to
    /// [`gemv_q1_g128_cached`](Self::gemv_q1_g128_cached) to perform GEMV
    /// without any host→device weight copy — **unless**
    /// [`weight_cache_enabled`](Self::weight_cache_enabled) is `false`
    /// (MET-M1), in which case the buffer is uploaded (so the caller's other
    /// side effects, e.g. a GPU warm-up pass, still run) but not retained,
    /// and the returned handle will not resolve.
    pub fn upload_weights(
        &self,
        block_bytes: &[u8],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        if !self.weight_cache_enabled.load(Ordering::Relaxed) {
            let _buf: GpuBuffer<u8> = self.ctx.create_buffer_from_slice(block_bytes);
            let id = NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed);
            debug!(
                handle = id,
                bytes = block_bytes.len(),
                "weight cache disabled (MET-M1): upload not retained"
            );
            return Ok(crate::weight_cache::GpuWeightHandle(id));
        }
        self.retain_upload(UploadLayout::RawBlocks, block_bytes)
    }

    /// Retain `stored` (already in its on-device layout) in the weight
    /// cache, sharing an existing byte-identical buffer when there is one.
    ///
    /// Two phases, so the potentially large byte comparison never runs under
    /// the cache lock: (1) look the content fingerprint up and, on a hit,
    /// compare the resident bytes outside the lock, then re-lock and register
    /// a reference only if that very buffer is still resident; (2) otherwise
    /// upload a fresh buffer under a new handle id. Either way the upload is
    /// attributed to the calling thread's [`super::current_upload_epoch`] and
    /// reported to the active [`super::GpuUploadScope`], so the engine can
    /// log exactly what it shared.
    fn retain_upload(
        &self,
        layout: UploadLayout,
        stored: &[u8],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        let (lo, hi) = content_fingerprint(stored);
        let content = ContentKey {
            layout,
            len: stored.len(),
            lo,
            hi,
        };
        let epoch = super::current_upload_epoch();

        // Phase 1: a byte-identical buffer may already be resident.
        let candidate = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| GpuError::NotAvailable("weight cache lock poisoned".into()))?;
            cache
                .by_content
                .get(&content)
                .and_then(|&handle| cache.get(handle).map(|buf| (handle, buf)))
        };
        if let Some((handle, resident)) = candidate {
            if resident_bytes_equal(&resident, stored) {
                let mut cache = self
                    .weight_cache
                    .lock()
                    .map_err(|_| GpuError::NotAvailable("weight cache lock poisoned".into()))?;
                let still_resident = cache
                    .entries
                    .get(&handle)
                    .is_some_and(|entry| entry.content == content);
                if still_resident {
                    cache.register(epoch, handle);
                    drop(cache);
                    super::note_weight_upload(stored.len() as u64, true);
                    debug!(
                        handle,
                        bytes = stored.len(),
                        epoch,
                        "weight upload shared an identical resident buffer"
                    );
                    return Ok(crate::weight_cache::GpuWeightHandle(handle));
                }
                // Released between the two phases: upload afresh below.
            } else {
                debug!(
                    handle,
                    bytes = stored.len(),
                    "content fingerprint matched but bytes differ; uploading a distinct buffer"
                );
            }
        }

        // Phase 2: a fresh buffer under a fresh handle id.
        let buf: GpuBuffer<u8> = self.ctx.create_buffer_from_slice(stored);
        let handle = NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed);
        {
            let mut cache = self
                .weight_cache
                .lock()
                .map_err(|_| GpuError::NotAvailable("weight cache lock poisoned".into()))?;
            cache.insert_fresh(epoch, handle, buf, content, stored.len());
        }
        super::note_weight_upload(stored.len() as u64, false);
        debug!(
            handle,
            bytes = stored.len(),
            epoch,
            "uploaded weights to GPU"
        );
        Ok(crate::weight_cache::GpuWeightHandle(handle))
    }

    /// Q1_0_g128 GEMV using a pre-uploaded weight buffer.
    ///
    /// Only the input vector is copied host→device (e.g. 4096 floats = 16 KB).
    /// The weight buffer is already GPU-resident.
    pub fn gemv_q1_g128_cached(
        &self,
        handle: crate::weight_cache::GpuWeightHandle,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        if k == 0 || !k.is_multiple_of(128) {
            return Err(GpuError::InvalidArgument(format!(
                "k={k} must be a positive multiple of 128"
            )));
        }
        if input.len() != k {
            return Err(GpuError::InvalidArgument(format!(
                "input.len()={} != k={}",
                input.len(),
                k
            )));
        }

        // Clone the cached GPU buffer (Arc clone, keeps GPU memory alive).
        let blocks_buf = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| GpuError::NotAvailable("weight cache lock poisoned".into()))?;
            cache.get(handle.0).ok_or_else(|| {
                GpuError::InvalidArgument(format!("invalid weight handle: {:?}", handle))
            })?
        };
        // Mutex is now released.

        let kernels = self.get_q1_kernels()?;

        let input_buf = self.ctx.create_buffer_from_slice(input);
        let output_buf: GpuBuffer<f32> = self.ctx.create_buffer(n_rows);

        // Per-call buffers, but `kernels.gemv` is still shared with
        // `gemv_q1_g128` / `gemm_q1_g128` / `batch_*_phase`, so the binding
        // and dispatch must take the same lock.
        let _dispatch = self.lock_dispatch()?;
        kernels.gemv.set_buffer("x", &blocks_buf);
        kernels.gemv.set_buffer("y", &input_buf);
        kernels.gemv.set_buffer("result", &output_buf);
        kernels.gemv.set_u32("n", n_rows as u32);
        kernels.gemv.set_u32("k", k as u32);

        kernels.gemv.dispatch([Self::workgroups_1d(n_rows), 1, 1]);

        Self::read_output(&output_buf, n_rows)
    }

    /// Number of cached weight entries currently resident on GPU.
    pub fn cached_weight_count(&self) -> usize {
        self.weight_cache
            .lock()
            .map(|c| c.entries.len())
            .unwrap_or(0)
    }

    /// Bytes of weight data currently resident in this backend's cache.
    pub fn cached_weight_bytes(&self) -> u64 {
        self.weight_cache
            .lock()
            .map(|c| c.resident_bytes)
            .unwrap_or(0)
    }

    /// Cumulative bytes this backend has **freshly** uploaded into its
    /// weight cache (a deduplicated upload adds nothing). Monotonic: not
    /// reduced by a release or a clear.
    pub fn uploaded_weight_bytes_total(&self) -> u64 {
        self.weight_cache
            .lock()
            .map(|c| c.uploaded_total)
            .unwrap_or(0)
    }

    /// Upload registrations `model_epoch` currently holds.
    pub fn epoch_registration_count(&self, model_epoch: u64) -> usize {
        self.weight_cache
            .lock()
            .map(|c| c.registrations(model_epoch))
            .unwrap_or(0)
    }

    /// Release one model epoch's registrations (`MET-M1`, eviction half):
    /// every buffer whose last reference belonged to `model_epoch` is freed
    /// — together with its `MetalGraph` mirror on a Metal build — and every
    /// buffer another epoch still references stays resident.
    ///
    /// This is what `InferenceEngine`'s `Drop` reaches (through
    /// [`super::release_model_weights`]); unlike
    /// [`clear_weight_cache`](Self::clear_weight_cache) it is safe with any
    /// number of live replicas, because a pool's replicas share buffers by
    /// reference, not by epoch. [`super::UNATTRIBUTED_MODEL_EPOCH`] is never
    /// released: its uploads have no known owner.
    ///
    /// Returns the number of buffers freed.
    ///
    /// # Errors
    ///
    /// [`GpuError::NotAvailable`] if the cache lock is poisoned.
    pub fn release_model_weights(&self, model_epoch: u64) -> Result<usize, GpuError> {
        if model_epoch == super::UNATTRIBUTED_MODEL_EPOCH {
            return Ok(0);
        }
        let (freed, remaining_bytes) = {
            let mut cache = self
                .weight_cache
                .lock()
                .map_err(|_| GpuError::NotAvailable("weight cache lock poisoned".into()))?;
            let freed = cache.release_epoch(model_epoch);
            (freed, cache.resident_bytes)
        };
        // The freed `GpuBuffer`s were dropped with their entries above; the
        // mirror eviction below only has ids to work with.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        release_metal_mirrors(&freed, model_epoch);
        let freed_bytes: usize = freed.iter().map(|&(_, bytes)| bytes).sum();
        debug!(
            model_epoch,
            freed_buffers = freed.len(),
            freed_bytes,
            remaining_bytes,
            "released a model epoch's GPU weights (MET-M1)"
        );
        Ok(freed.len())
    }

    /// Evict every cached weight buffer, freeing its GPU memory.
    ///
    /// **Process-wide.** Every epoch's registrations go with it, so this must
    /// not be called while any engine that uploaded into this backend is
    /// still alive — use [`release_model_weights`](Self::release_model_weights)
    /// for one model. Kept for the moment the fused `MetalGraph` decode path
    /// (which maintains its own, separately-keyed weight cache — see MET-02)
    /// takes over for a whole process. Combine with
    /// [`set_weight_cache_enabled(false)`](Self::set_weight_cache_enabled)
    /// to also stop new uploads from being retained.
    ///
    /// Returns the number of entries evicted.
    pub fn clear_weight_cache(&self) -> Result<usize, GpuError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| GpuError::NotAvailable("weight cache lock poisoned".into()))?;
        let evicted = cache.clear();
        debug!(evicted, "cleared Scirs2Backend GPU weight cache (MET-M1)");
        Ok(evicted)
    }

    /// Upload TQ2_0_g128 (ternary) weight blocks to GPU in SoA layout.
    ///
    /// Converts AoS block layout `[qs: [u8;32], d: f16]` (34 bytes/block) to
    /// SoA layout `[N×2B FP16 scales][N×32B qs data]` for coalesced GPU reads.
    ///
    /// Returns a [`GpuWeightHandle`](crate::weight_cache::GpuWeightHandle) for use with
    /// [`gemv_tq2_g128_cached`](Self::gemv_tq2_g128_cached) — **unless**
    /// [`weight_cache_enabled`](Self::weight_cache_enabled) is `false`
    /// (MET-M1), in which case the buffer is uploaded but not retained, and
    /// the returned handle will not resolve.
    pub fn upload_weights_ternary(
        &self,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        let n = blocks.len();
        // SoA layout: [n * 2 bytes (FP16 scales LE)] ++ [n * 32 bytes (qs data)]
        let mut soa = Vec::with_capacity(n * 34);
        // Phase 1: all FP16 scales (2 bytes each, little-endian)
        for block in blocks {
            let bits = block.d.to_bits().to_le_bytes();
            soa.push(bits[0]);
            soa.push(bits[1]);
        }
        // Phase 2: all qs arrays (32 bytes each)
        for block in blocks {
            soa.extend_from_slice(&block.qs);
        }
        if !self.weight_cache_enabled.load(Ordering::Relaxed) {
            let _buf: scirs2_core::gpu::GpuBuffer<u8> = self.ctx.create_buffer_from_slice(&soa);
            let id = NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed);
            debug!(
                handle = id,
                blocks = n,
                bytes = soa.len(),
                "weight cache disabled (MET-M1): ternary upload not retained"
            );
            return Ok(crate::weight_cache::GpuWeightHandle(id));
        }
        self.retain_upload(UploadLayout::TernarySoa, &soa)
    }

    /// TQ2_0_g128 (ternary) GEMV using a pre-uploaded SoA weight buffer.
    ///
    /// Only the input vector is copied host→device.
    /// The weight buffer (SoA layout) is already GPU-resident.
    pub fn gemv_tq2_g128_cached(
        &self,
        handle: crate::weight_cache::GpuWeightHandle,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        if k == 0 || !k.is_multiple_of(128) {
            return Err(GpuError::InvalidArgument(format!(
                "k={k} must be a positive multiple of 128"
            )));
        }
        if input.len() != k {
            return Err(GpuError::InvalidArgument(format!(
                "input.len()={} != k={}",
                input.len(),
                k
            )));
        }

        // Clone the cached GPU buffer (Arc clone, keeps GPU memory alive).
        let blocks_buf = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| GpuError::NotAvailable("weight cache lock poisoned".into()))?;
            cache.get(handle.0).ok_or_else(|| {
                GpuError::InvalidArgument(format!("invalid ternary weight handle: {:?}", handle))
            })?
        };
        // Mutex is now released.

        let tq2 = self.get_tq2_kernels()?;

        // Input must be reinterpreted as float4 in the MSL kernel (buffer(1) is float4*).
        // We pass raw f32 bytes; MSL will read them as float4.
        let input_buf = self.ctx.create_buffer_from_slice(input);
        let output_buf: scirs2_core::gpu::GpuBuffer<f32> = self.ctx.create_buffer(n_rows);

        let _dispatch = self.lock_dispatch()?;
        tq2.gemv.set_buffer("x", &blocks_buf);
        tq2.gemv.set_buffer("y", &input_buf);
        tq2.gemv.set_buffer("result", &output_buf);
        tq2.gemv.set_u32("n", n_rows as u32);
        tq2.gemv.set_u32("k", k as u32);

        tq2.gemv.dispatch([Self::workgroups_simd(n_rows), 1, 1]);

        Self::read_output(&output_buf, n_rows)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Pipeline GPU dispatch (dispatch + final dispatch with wait)
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
impl Scirs2Backend {
    /// Get a cached weight buffer by handle.
    fn get_cached_weight(
        &self,
        handle: crate::weight_cache::GpuWeightHandle,
    ) -> Result<GpuBuffer<u8>, GpuError> {
        let cache = self
            .weight_cache
            .lock()
            .map_err(|_| GpuError::NotAvailable("weight cache lock poisoned".into()))?;
        cache.get(handle.0).ok_or_else(|| {
            GpuError::InvalidArgument(format!("invalid weight handle: {:?}", handle))
        })
    }

    /// Lazily allocate pipeline buffers for the FFN phase.
    fn get_or_init_pipeline_buffers(
        &self,
        h: usize,
        attn_out_size: usize,
        intermediate: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<PipelineBuffers>>, GpuError> {
        let mut guard = self
            .pipeline_buffers
            .lock()
            .map_err(|_| GpuError::NotAvailable("pipeline buffers lock poisoned".into()))?;
        if guard.is_none() {
            info!(
                h,
                attn_out_size, intermediate, "allocating pipeline GPU buffers"
            );
            *guard = Some(PipelineBuffers {
                hidden: self.ctx.create_buffer(h),
                attn_out: self.ctx.create_buffer(attn_out_size),
                norm_weight: self.ctx.create_buffer(h),
                attn_proj: self.ctx.create_buffer(h),
                normed: self.ctx.create_buffer(h),
                gate_up: self.ctx.create_buffer(intermediate * 2),
                swiglu: self.ctx.create_buffer(intermediate),
                down: self.ctx.create_buffer(h),
            });
        }
        Ok(guard)
    }

    /// RMSNorm + fused QKV GEMV using dispatch pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_attn_phase(
        &self,
        hidden: &[f32],
        norm_weight: &[f32],
        norm_eps: f32,
        qkv_handle: crate::weight_cache::GpuWeightHandle,
        q_rows: usize,
        k_rows: usize,
        h: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), GpuError> {
        #![allow(clippy::type_complexity)]
        let q1 = self.get_q1_kernels()?;
        let helpers = self.get_helper_kernels()?;
        let qkv_weights = self.get_cached_weight(qkv_handle)?;
        let total_rows = q_rows + k_rows + k_rows;

        let hidden_buf = self.ctx.create_buffer_from_slice(hidden);
        let norm_buf = self.ctx.create_buffer_from_slice(norm_weight);
        let normed_buf: GpuBuffer<f32> = self.ctx.create_buffer(h);
        let qkv_buf: GpuBuffer<f32> = self.ctx.create_buffer(total_rows);

        // One critical section for the whole two-kernel sequence: the
        // intermediate `normed_buf` is produced by the first dispatch and
        // consumed by the second, so another caller rebinding either kernel
        // in between would corrupt both.
        let _dispatch = self.lock_dispatch()?;

        // RMSNorm (no wait)
        helpers.rmsnorm_weighted.set_buffer("x", &hidden_buf);
        helpers.rmsnorm_weighted.set_buffer("y", &norm_buf);
        helpers.rmsnorm_weighted.set_buffer("result", &normed_buf);
        helpers.rmsnorm_weighted.set_u32("n", h as u32);
        helpers.rmsnorm_weighted.set_f32("alpha", norm_eps);
        helpers
            .rmsnorm_weighted
            .dispatch([Self::workgroups_1d(h), 1, 1]);

        // QKV GEMV (wait — Metal queue guarantees order)
        q1.gemv.set_buffer("x", &qkv_weights);
        q1.gemv.set_buffer("y", &normed_buf);
        q1.gemv.set_buffer("result", &qkv_buf);
        q1.gemv.set_u32("n", total_rows as u32);
        q1.gemv.set_u32("k", h as u32);
        q1.gemv.dispatch([Self::workgroups_simd(total_rows), 1, 1]);

        let qkv_data = Self::read_output(&qkv_buf, total_rows)?;
        Ok((
            qkv_data[..q_rows].to_vec(),
            qkv_data[q_rows..q_rows + k_rows].to_vec(),
            qkv_data[q_rows + k_rows..total_rows].to_vec(),
        ))
    }

    /// Pipeline the entire FFN phase using dispatch.
    ///
    /// Uses 6 x dispatch + 1 x dispatch (with wait).
    /// Metal command queue guarantees in-order execution.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_ffn_phase(
        &self,
        hidden: &mut [f32],
        attn_out: &[f32],
        norm_weight: &[f32],
        norm_eps: f32,
        attn_proj_handle: crate::weight_cache::GpuWeightHandle,
        gate_up_handle: crate::weight_cache::GpuWeightHandle,
        down_handle: crate::weight_cache::GpuWeightHandle,
        h: usize,
        intermediate: usize,
        attn_proj_k: usize,
    ) -> Result<(), GpuError> {
        let q1 = self.get_q1_kernels()?;
        let helpers = self.get_helper_kernels()?;

        let attn_proj_w = self.get_cached_weight(attn_proj_handle)?;
        let gate_up_w = self.get_cached_weight(gate_up_handle)?;
        let down_w = self.get_cached_weight(down_handle)?;

        // Clone Arc references from pre-allocated buffers (zero allocation)
        let (
            hidden_buf,
            attn_out_buf,
            norm_buf,
            attn_proj_buf,
            normed_buf,
            gate_up_buf,
            swiglu_buf,
            down_buf,
        ) = {
            let guard = self.get_or_init_pipeline_buffers(h, attn_proj_k, intermediate)?;
            let bufs = guard
                .as_ref()
                .ok_or_else(|| GpuError::NotAvailable("pipeline buffers not initialized".into()))?;
            (
                bufs.hidden.clone(),
                bufs.attn_out.clone(),
                bufs.norm_weight.clone(),
                bufs.attn_proj.clone(),
                bufs.normed.clone(),
                bufs.gate_up.clone(),
                bufs.swiglu.clone(),
                bufs.down.clone(),
            )
        };

        // The `pipeline_buffers` guard is released above; take the dispatch
        // lock only now, so the lock order is always
        // `pipeline_buffers` → (released) → `dispatch`. It covers the whole
        // seven-step pipeline, whose intermediates live in those shared
        // pipeline buffers.
        let _dispatch = self.lock_dispatch()?;

        // Copy input data (memcpy to shared memory, no allocation)
        hidden_buf
            .copy_from_host(hidden)
            .map_err(|e| Self::convert_error(&e))?;
        attn_out_buf
            .copy_from_host(attn_out)
            .map_err(|e| Self::convert_error(&e))?;
        norm_buf
            .copy_from_host(norm_weight)
            .map_err(|e| Self::convert_error(&e))?;

        // 1. attn_proj GEMV (no wait)
        q1.gemv.set_buffer("x", &attn_proj_w);
        q1.gemv.set_buffer("y", &attn_out_buf);
        q1.gemv.set_buffer("result", &attn_proj_buf);
        q1.gemv.set_u32("n", h as u32);
        q1.gemv.set_u32("k", attn_proj_k as u32);
        q1.gemv.dispatch([Self::workgroups_simd(h), 1, 1]);

        // 2. residual_add: hidden += attn_proj (no wait)
        helpers.residual_add.set_buffer("x", &hidden_buf);
        helpers.residual_add.set_buffer("y", &attn_proj_buf);
        helpers.residual_add.set_u32("n", h as u32);
        helpers
            .residual_add
            .dispatch([Self::workgroups_1d(h), 1, 1]);

        // 3. FFN RMSNorm: hidden -> normed (no wait)
        helpers.rmsnorm_weighted.set_buffer("x", &hidden_buf);
        helpers.rmsnorm_weighted.set_buffer("y", &norm_buf);
        helpers.rmsnorm_weighted.set_buffer("result", &normed_buf);
        helpers.rmsnorm_weighted.set_u32("n", h as u32);
        helpers.rmsnorm_weighted.set_f32("alpha", norm_eps);
        helpers
            .rmsnorm_weighted
            .dispatch([Self::workgroups_1d(h), 1, 1]);

        // 4. fused gate_up GEMV (no wait)
        q1.gemv.set_buffer("x", &gate_up_w);
        q1.gemv.set_buffer("y", &normed_buf);
        q1.gemv.set_buffer("result", &gate_up_buf);
        q1.gemv.set_u32("n", (intermediate * 2) as u32);
        q1.gemv.set_u32("k", h as u32);
        q1.gemv
            .dispatch([Self::workgroups_simd(intermediate * 2), 1, 1]);

        // 5. SwiGLU (no wait)
        helpers.swiglu_fused.set_buffer("x", &gate_up_buf);
        helpers.swiglu_fused.set_buffer("result", &swiglu_buf);
        helpers.swiglu_fused.set_u32("n", intermediate as u32);
        helpers
            .swiglu_fused
            .dispatch([Self::workgroups_1d(intermediate), 1, 1]);

        // 6. down GEMV (no wait)
        q1.gemv.set_buffer("x", &down_w);
        q1.gemv.set_buffer("y", &swiglu_buf);
        q1.gemv.set_buffer("result", &down_buf);
        q1.gemv.set_u32("n", h as u32);
        q1.gemv.set_u32("k", intermediate as u32);
        q1.gemv.dispatch([Self::workgroups_simd(h), 1, 1]);

        // 7. residual_add: hidden += down (WAIT — ensures all above complete)
        helpers.residual_add.set_buffer("x", &hidden_buf);
        helpers.residual_add.set_buffer("y", &down_buf);
        helpers.residual_add.set_u32("n", h as u32);
        helpers
            .residual_add
            .dispatch([Self::workgroups_1d(h), 1, 1]);

        // Read back modified hidden state
        hidden_buf
            .copy_to_host(hidden)
            .map_err(|e| Self::convert_error(&e))?;
        Ok(())
    }

    /// Expose the GPU context for external buffer creation.
    pub fn gpu_context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Create a GPU buffer from a slice.
    pub fn create_buffer_from_slice<T: GpuDataType>(&self, data: &[T]) -> GpuBuffer<T> {
        self.ctx.create_buffer_from_slice(data)
    }

    /// Create an empty GPU buffer.
    pub fn create_buffer<T: GpuDataType>(&self, len: usize) -> GpuBuffer<T> {
        self.ctx.create_buffer(len)
    }

    /// Upload raw weight bytes and return a handle.
    pub fn upload_weights_raw(
        &self,
        bytes: &[u8],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        self.upload_weights(bytes)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────

// The unit tests live in a sibling file so this module stays well inside the
// 2000-line policy ceiling (it was at 1955 before `MET-08`). `#[path]` keeps
// them a child module of `scirs2_backend`, so they still reach its private
// items through `use super::*` — the same shape `attention_fused.rs` uses.
#[cfg(test)]
#[cfg(feature = "gpu")]
#[path = "scirs2_backend_tests.rs"]
mod tests;
