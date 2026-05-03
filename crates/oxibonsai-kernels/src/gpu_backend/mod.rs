//! GPU backend abstraction layer for CUDA and Metal acceleration.
//!
//! This module defines the [`GpuBackendTrait`] trait and provides:
//! - [`CpuBackend`]: Always-available CPU implementation (baseline)
//! - [`CudaBackend`]: CUDA stub (feature = "cuda", compile-only placeholder)
//! - [`MetalBackend`]: Metal stub (feature = "metal", compile-only placeholder)
//! - [`Scirs2Backend`]: **Real** GPU backend via scirs2-core (feature = "gpu")
//!
//! # Architecture
//! All GPU operations follow the same pattern:
//! 1. Allocate device buffers
//! 2. Copy host → device
//! 3. Execute kernel
//! 4. Copy device → host
//!
//! The [`Scirs2Backend`] compiles Metal/CUDA kernels at runtime through
//! scirs2-core and dispatches real GPU work.  Stub backends delegate to
//! CPU operations.
//!
//! # Q1_0_g128 GPU acceleration
//!
//! The [`gpu_gemv_1bit`] function provides a high-level entry point for
//! 1-bit quantised matrix-vector multiplication on the GPU.

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_attn_kernels;
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_full_layer;
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_graph;
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_kernels;
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_prefill;
#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub mod cuda_prefill_kernels;
pub mod kernel_sources;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_dispatch;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_full_layer;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_graph;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_prefill;
pub mod scirs2_backend;

use thiserror::Error;
#[allow(unused_imports)]
use tracing::warn;

#[cfg(feature = "gpu")]
pub use scirs2_backend::Scirs2Backend;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal_graph::{MetalGraph, MetalGraphError, MetalWeightHandle};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal_full_layer::{
    build_cached_weights, build_cached_weights_ternary_only, print_gpu_profile_summary,
    try_metal_ffn, try_metal_forward_greedy_ternary, try_metal_full_forward,
    try_metal_full_forward_cached, try_metal_full_forward_ternary, try_metal_full_layer,
    try_metal_prefill_ternary, try_metal_prefill_verify_ternary, try_metal_qkv, CachedLayerWeights,
    CachedModelWeights, FullForwardLayerParams, FullForwardLayerParamsTernary,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal_prefill::{try_metal_full_forward_prefill, try_metal_full_forward_prefill_verify};

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub use cuda_graph::{try_cuda_ffn, try_cuda_qkv, CudaGraph, CudaGraphError, NativeCudaBackend};

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub use cuda_full_layer::{
    try_cuda_full_forward, try_cuda_full_forward_ternary,
    try_cuda_full_forward_ternary_with_gpu_lm_head, try_cuda_full_forward_with_gpu_lm_head,
    try_cuda_full_layer, CudaCachedLayerWeights, CudaFullForwardLayerParams,
    CudaFullForwardLayerParamsTernary,
};

#[cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]
pub use cuda_prefill::try_cuda_prefill;

// ═══════════════════════════════════════════════════════════════════════════
// DeviceBuffer
// ═══════════════════════════════════════════════════════════════════════════

/// Device memory buffer (opaque handle).
///
/// For CPU and stub backends this is simply a heap-allocated `Vec<f32>`.
/// A future hardware backend would replace `data` with a raw device pointer
/// and keep the `Vec` only as a host-side staging buffer.
pub struct DeviceBuffer {
    /// CPU backing store (used by all stub backends).
    pub data: Vec<f32>,
    /// Number of `f32` elements in the buffer.
    pub size: usize,
    /// Logical device index this buffer is associated with.
    pub device_id: usize,
}

impl DeviceBuffer {
    /// Allocate a zero-initialised buffer of `size` elements on `device_id`.
    pub fn new(size: usize, device_id: usize) -> Self {
        Self {
            data: vec![0.0_f32; size],
            size,
            device_id,
        }
    }

    /// Create a buffer pre-populated from a host slice.
    pub fn from_slice(data: &[f32], device_id: usize) -> Self {
        let size = data.len();
        Self {
            data: data.to_vec(),
            size,
            device_id,
        }
    }

    /// Copy the buffer contents back to a host `Vec<f32>`.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Number of `f32` elements stored in this buffer.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Logical device index this buffer is bound to.
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LaunchConfig
// ═══════════════════════════════════════════════════════════════════════════

/// Kernel launch configuration (CUDA-style grid/block decomposition).
///
/// For CPU and Metal backends these values are informational only; the actual
/// parallelism strategy is determined by the backend itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Number of thread-blocks (x, y, z).
    pub grid_dim: (u32, u32, u32),
    /// Threads per block (x, y, z).
    pub block_dim: (u32, u32, u32),
    /// Dynamic shared memory per block in bytes.
    pub shared_mem_bytes: u32,
}

/// Default block size used by `for_n_elements`.
const DEFAULT_BLOCK_SIZE: u32 = 256;

impl LaunchConfig {
    /// Auto-compute a 1-D launch configuration for `n` elements.
    ///
    /// Uses a block size of 256 threads and rounds the grid up to cover all
    /// elements.  `shared_mem_bytes` is set to zero.
    pub fn for_n_elements(n: usize) -> Self {
        let block = DEFAULT_BLOCK_SIZE;
        let grid = ((n as u32).saturating_add(block - 1)) / block;
        Self {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// A sensible default 1-D config (1 block of 256 threads).
    pub fn default_1d() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (DEFAULT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GpuError
// ═══════════════════════════════════════════════════════════════════════════

/// Error type for GPU backend operations.
#[derive(Debug, Error)]
pub enum GpuError {
    /// The requested GPU/backend is not present or not compiled in.
    #[error("GPU not available: {0}")]
    NotAvailable(String),

    /// A device-side allocation failed due to insufficient memory.
    #[error("out of device memory: requested {requested} bytes on device {device}")]
    OutOfMemory {
        /// Requested allocation size in bytes.
        requested: usize,
        /// Device index that was targeted.
        device: usize,
    },

    /// A kernel could not be launched (bad dimensions, missing module, etc.).
    #[error("kernel launch failed: {0}")]
    KernelLaunch(String),

    /// The device failed to synchronise after kernel execution.
    #[error("device synchronization failed: {0}")]
    SyncFailed(String),

    /// A parameter value is out of range or logically inconsistent.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

// ═══════════════════════════════════════════════════════════════════════════
// GpuBackendTrait
// ═══════════════════════════════════════════════════════════════════════════

/// Core GPU backend trait.
///
/// Implementations of this trait provide the primitive operations required by
/// the OxiBonsai inference engine.  The [`CpuBackend`] is always available
/// and is used as a correctness baseline; hardware backends are feature-gated.
///
/// # Backwards compatibility
///
/// This trait was previously named `GpuBackend`.  The type alias
/// [`GpuBackend`] preserves source compatibility.
pub trait GpuBackendTrait: Send + Sync {
    /// Human-readable backend identifier (e.g. `"cpu"`, `"cuda"`, `"metal"`).
    fn name(&self) -> &'static str;

    /// Returns `true` only when the backend is backed by real GPU hardware.
    fn is_accelerated(&self) -> bool;

    /// Number of logical devices available to this backend.
    fn device_count(&self) -> usize;

    /// Allocate an uninitialised (zero-filled for stubs) device buffer.
    fn alloc(&self, size: usize, device_id: usize) -> Result<DeviceBuffer, GpuError>;

    /// Copy a host slice to a new device buffer and return it.
    fn host_to_device(&self, src: &[f32], device_id: usize) -> Result<DeviceBuffer, GpuError>;

    /// Copy a device buffer to a new host `Vec<f32>`.
    fn device_to_host(&self, buf: &DeviceBuffer) -> Result<Vec<f32>, GpuError>;

    /// Matrix-vector multiply: **y = A · x**.
    ///
    /// - `a` — row-major matrix of shape `[m, k]`
    /// - `x` — column vector of length `k`
    /// - Returns a buffer of length `m`.
    fn matvec(
        &self,
        a: &DeviceBuffer,
        x: &DeviceBuffer,
        m: usize,
        k: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError>;

    /// Element-wise ReLU: **y_i = max(0, x_i)**.
    fn relu(&self, x: &DeviceBuffer, device_id: usize) -> Result<DeviceBuffer, GpuError>;

    /// Softmax over the entire buffer (treated as a 1-D vector of `size` elements).
    fn softmax(
        &self,
        x: &DeviceBuffer,
        size: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError>;

    /// Block until all previously submitted kernels on `device_id` have finished.
    fn synchronize(&self, device_id: usize) -> Result<(), GpuError>;

    /// Query device memory: returns `(free_bytes, total_bytes)`.
    fn memory_info(&self, device_id: usize) -> Result<(usize, usize), GpuError>;

    /// Q1_0_g128 matrix-vector product.
    ///
    /// Default implementation falls back to CPU dequant + scalar GEMV.
    /// [`Scirs2Backend`] overrides this with a real GPU kernel.
    fn gemv_q1_g128(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        cpu_gemv_1bit_fallback(block_bytes, input, n_rows, k)
    }

    /// Q1_0_g128 matrix-matrix product.
    ///
    /// Default implementation falls back to repeated [`gemv_q1_g128`](Self::gemv_q1_g128) calls.
    fn gemm_q1_g128(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let mut output = vec![0.0_f32; m * n_rows];
        for i in 0..m {
            let row_input = &input[i * k..(i + 1) * k];
            let row_output = self.gemv_q1_g128(block_bytes, row_input, n_rows, k)?;
            output[i * n_rows..(i + 1) * n_rows].copy_from_slice(&row_output);
        }
        Ok(output)
    }

    /// Upload weight block bytes to GPU memory and return a reusable handle.
    ///
    /// Default: not supported (returns `NotAvailable`).
    fn upload_weights_raw(
        &self,
        _block_bytes: &[u8],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        Err(GpuError::NotAvailable(
            "weight caching not supported by this backend".into(),
        ))
    }

    /// Q1_0_g128 GEMV using a pre-uploaded GPU-resident weight buffer.
    ///
    /// Default: not supported (returns `NotAvailable`).
    fn gemv_q1_g128_cached(
        &self,
        _handle: crate::weight_cache::GpuWeightHandle,
        _input: &[f32],
        _n_rows: usize,
        _k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        Err(GpuError::NotAvailable(
            "cached GEMV not supported by this backend".into(),
        ))
    }

    /// Upload TQ2_0_g128 weight blocks to GPU memory in SoA layout.
    ///
    /// Default: not supported (returns `NotAvailable`).
    fn upload_weights_ternary(
        &self,
        _blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        Err(GpuError::NotAvailable(
            "ternary weight upload not supported by this backend".into(),
        ))
    }

    /// TQ2_0_g128 GEMV using a pre-uploaded GPU-resident weight buffer.
    ///
    /// Default: not supported (returns `NotAvailable`).
    fn gemv_tq2_g128_cached(
        &self,
        _handle: crate::weight_cache::GpuWeightHandle,
        _input: &[f32],
        _n_rows: usize,
        _k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        Err(GpuError::NotAvailable(
            "cached ternary GEMV not supported by this backend".into(),
        ))
    }

    /// Batch-execute attention input phase (RMSNorm + QKV) in one command buffer.
    ///
    /// Returns `Ok(Some((q, k, v)))` if batching succeeded, or `Ok(None)` if
    /// not supported by this backend.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn batch_attn_phase(
        &self,
        _hidden: &[f32],
        _norm_weight: &[f32],
        _norm_eps: f32,
        _qkv_handle: crate::weight_cache::GpuWeightHandle,
        _q_rows: usize,
        _k_rows: usize,
        _h: usize,
    ) -> Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>, GpuError> {
        Ok(None)
    }

    /// Batch-execute FFN phase in one command buffer.
    ///
    /// Returns `Ok(true)` if batching succeeded and `hidden` was modified
    /// in-place, or `Ok(false)` if not supported by this backend.
    #[allow(clippy::too_many_arguments)]
    fn batch_ffn_phase(
        &self,
        _hidden: &mut [f32],
        _attn_out: &[f32],
        _norm_weight: &[f32],
        _norm_eps: f32,
        _attn_proj_handle: crate::weight_cache::GpuWeightHandle,
        _gate_up_handle: crate::weight_cache::GpuWeightHandle,
        _down_handle: crate::weight_cache::GpuWeightHandle,
        _h: usize,
        _intermediate: usize,
        _attn_proj_k: usize,
    ) -> Result<bool, GpuError> {
        Ok(false)
    }
}

/// Backwards-compatible type alias for the GPU backend trait.
///
/// Existing code that references `GpuBackend` as a trait will continue to
/// compile.
pub type GpuBackend = dyn GpuBackendTrait;

// ═══════════════════════════════════════════════════════════════════════════
// CpuBackend
// ═══════════════════════════════════════════════════════════════════════════

/// CPU backend — always available, no GPU required.
///
/// Implements [`GpuBackendTrait`] using plain scalar Rust operations.
pub struct CpuBackend {
    /// Simulated total device memory reported by `memory_info`.
    pub simulated_memory_bytes: usize,
}

impl CpuBackend {
    /// Create a `CpuBackend` with a default simulated memory of 4 GiB.
    pub fn new() -> Self {
        Self {
            simulated_memory_bytes: 4 * 1024 * 1024 * 1024,
        }
    }

    /// Create a `CpuBackend` with a custom simulated memory size (bytes).
    pub fn with_memory(bytes: usize) -> Self {
        Self {
            simulated_memory_bytes: bytes,
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackendTrait for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn is_accelerated(&self) -> bool {
        false
    }

    fn device_count(&self) -> usize {
        1
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

        let mut result = vec![0.0_f32; m];
        for (row, slot) in result.iter_mut().enumerate().take(m) {
            let mut acc = 0.0_f32;
            for col in 0..k {
                acc += a.data[row * k + col] * x.data[col];
            }
            *slot = acc;
        }

        Ok(DeviceBuffer::from_slice(&result, device_id))
    }

    fn relu(&self, x: &DeviceBuffer, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        let result: Vec<f32> = x.data.iter().map(|&v| v.max(0.0)).collect();
        Ok(DeviceBuffer::from_slice(&result, device_id))
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

    fn synchronize(&self, _device_id: usize) -> Result<(), GpuError> {
        Ok(())
    }

    fn memory_info(&self, _device_id: usize) -> Result<(usize, usize), GpuError> {
        let total = self.simulated_memory_bytes;
        let free = total / 2;
        Ok((free, total))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CudaBackend (stub, feature = "cuda")
// ═══════════════════════════════════════════════════════════════════════════

/// CUDA backend stub — feature-gated, compile-only placeholder.
///
/// All operations delegate to `CpuBackend` and emit a `warn!` trace event.
/// Use [`Scirs2Backend`] for real GPU acceleration.
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    /// Number of CUDA devices detected at construction time.
    pub device_count: usize,
    cpu_fallback: CpuBackend,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Attempt to initialise the CUDA backend (stub).
    pub fn new() -> Result<Self, GpuError> {
        warn!("CudaBackend: CUDA stub active — no real GPU acceleration");
        Ok(Self {
            device_count: 1,
            cpu_fallback: CpuBackend::new(),
        })
    }
}

#[cfg(feature = "cuda")]
impl GpuBackendTrait for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn is_accelerated(&self) -> bool {
        false
    }

    fn device_count(&self) -> usize {
        self.device_count
    }

    fn alloc(&self, size: usize, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        warn!("CudaBackend::alloc delegating to CPU fallback");
        self.cpu_fallback.alloc(size, device_id)
    }

    fn host_to_device(&self, src: &[f32], device_id: usize) -> Result<DeviceBuffer, GpuError> {
        warn!("CudaBackend::host_to_device delegating to CPU fallback");
        self.cpu_fallback.host_to_device(src, device_id)
    }

    fn device_to_host(&self, buf: &DeviceBuffer) -> Result<Vec<f32>, GpuError> {
        warn!("CudaBackend::device_to_host delegating to CPU fallback");
        self.cpu_fallback.device_to_host(buf)
    }

    fn matvec(
        &self,
        a: &DeviceBuffer,
        x: &DeviceBuffer,
        m: usize,
        k: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        warn!("CudaBackend::matvec delegating to CPU fallback");
        self.cpu_fallback.matvec(a, x, m, k, device_id)
    }

    fn relu(&self, x: &DeviceBuffer, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        warn!("CudaBackend::relu delegating to CPU fallback");
        self.cpu_fallback.relu(x, device_id)
    }

    fn softmax(
        &self,
        x: &DeviceBuffer,
        size: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        warn!("CudaBackend::softmax delegating to CPU fallback");
        self.cpu_fallback.softmax(x, size, device_id)
    }

    fn synchronize(&self, device_id: usize) -> Result<(), GpuError> {
        warn!("CudaBackend::synchronize delegating to CPU fallback");
        self.cpu_fallback.synchronize(device_id)
    }

    fn memory_info(&self, device_id: usize) -> Result<(usize, usize), GpuError> {
        warn!("CudaBackend::memory_info delegating to CPU fallback");
        self.cpu_fallback.memory_info(device_id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MetalBackend (stub, feature = "metal", macOS only)
// ═══════════════════════════════════════════════════════════════════════════

/// Metal backend stub — feature-gated, macOS only, compile-only placeholder.
///
/// Use [`Scirs2Backend`] for real GPU acceleration.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct MetalBackend {
    /// Number of Metal devices detected at construction time.
    pub device_count: usize,
    cpu_fallback: CpuBackend,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl MetalBackend {
    /// Attempt to initialise the Metal backend (stub).
    pub fn new() -> Result<Self, GpuError> {
        warn!("MetalBackend: Metal stub active — no real GPU acceleration");
        Ok(Self {
            device_count: 1,
            cpu_fallback: CpuBackend::new(),
        })
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl GpuBackendTrait for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn is_accelerated(&self) -> bool {
        false
    }

    fn device_count(&self) -> usize {
        self.device_count
    }

    fn alloc(&self, size: usize, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        warn!("MetalBackend::alloc delegating to CPU fallback");
        self.cpu_fallback.alloc(size, device_id)
    }

    fn host_to_device(&self, src: &[f32], device_id: usize) -> Result<DeviceBuffer, GpuError> {
        warn!("MetalBackend::host_to_device delegating to CPU fallback");
        self.cpu_fallback.host_to_device(src, device_id)
    }

    fn device_to_host(&self, buf: &DeviceBuffer) -> Result<Vec<f32>, GpuError> {
        warn!("MetalBackend::device_to_host delegating to CPU fallback");
        self.cpu_fallback.device_to_host(buf)
    }

    fn matvec(
        &self,
        a: &DeviceBuffer,
        x: &DeviceBuffer,
        m: usize,
        k: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        warn!("MetalBackend::matvec delegating to CPU fallback");
        self.cpu_fallback.matvec(a, x, m, k, device_id)
    }

    fn relu(&self, x: &DeviceBuffer, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        warn!("MetalBackend::relu delegating to CPU fallback");
        self.cpu_fallback.relu(x, device_id)
    }

    fn softmax(
        &self,
        x: &DeviceBuffer,
        size: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        warn!("MetalBackend::softmax delegating to CPU fallback");
        self.cpu_fallback.softmax(x, size, device_id)
    }

    fn synchronize(&self, device_id: usize) -> Result<(), GpuError> {
        warn!("MetalBackend::synchronize delegating to CPU fallback");
        self.cpu_fallback.synchronize(device_id)
    }

    fn memory_info(&self, device_id: usize) -> Result<(usize, usize), GpuError> {
        warn!("MetalBackend::memory_info delegating to CPU fallback");
        self.cpu_fallback.memory_info(device_id)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scirs2BackendHandle (singleton wrapper)
// ═══════════════════════════════════════════════════════════════════════════

/// Thin wrapper around `Arc<Scirs2Backend>` that implements [`GpuBackendTrait`].
///
/// This allows the process-wide singleton to be used wherever a
/// `Box<dyn GpuBackendTrait>` is expected (e.g. [`select_backend`]).
#[cfg(feature = "gpu")]
pub(crate) struct Scirs2BackendHandle(pub(crate) std::sync::Arc<Scirs2Backend>);

#[cfg(feature = "gpu")]
impl GpuBackendTrait for Scirs2BackendHandle {
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn is_accelerated(&self) -> bool {
        self.0.is_accelerated()
    }
    fn device_count(&self) -> usize {
        self.0.device_count()
    }
    fn alloc(&self, size: usize, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        self.0.alloc(size, device_id)
    }
    fn host_to_device(&self, src: &[f32], device_id: usize) -> Result<DeviceBuffer, GpuError> {
        self.0.host_to_device(src, device_id)
    }
    fn device_to_host(&self, buf: &DeviceBuffer) -> Result<Vec<f32>, GpuError> {
        self.0.device_to_host(buf)
    }
    fn matvec(
        &self,
        a: &DeviceBuffer,
        x: &DeviceBuffer,
        m: usize,
        k: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        self.0.matvec(a, x, m, k, device_id)
    }
    fn relu(&self, x: &DeviceBuffer, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        self.0.relu(x, device_id)
    }
    fn softmax(
        &self,
        x: &DeviceBuffer,
        size: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        self.0.softmax(x, size, device_id)
    }
    fn synchronize(&self, device_id: usize) -> Result<(), GpuError> {
        self.0.synchronize(device_id)
    }
    fn memory_info(&self, device_id: usize) -> Result<(usize, usize), GpuError> {
        self.0.memory_info(device_id)
    }
    fn gemv_q1_g128(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.0.gemv_q1_g128(block_bytes, input, n_rows, k)
    }
    fn gemm_q1_g128(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        m: usize,
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.0.gemm_q1_g128(block_bytes, input, m, n_rows, k)
    }
    fn upload_weights_raw(
        &self,
        block_bytes: &[u8],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        self.0.upload_weights(block_bytes)
    }
    fn gemv_q1_g128_cached(
        &self,
        handle: crate::weight_cache::GpuWeightHandle,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.0.gemv_q1_g128_cached(handle, input, n_rows, k)
    }

    fn upload_weights_ternary(
        &self,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        self.0.upload_weights_ternary(blocks)
    }

    fn gemv_tq2_g128_cached(
        &self,
        handle: crate::weight_cache::GpuWeightHandle,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.0.gemv_tq2_g128_cached(handle, input, n_rows, k)
    }

    fn batch_attn_phase(
        &self,
        hidden: &[f32],
        norm_weight: &[f32],
        norm_eps: f32,
        qkv_handle: crate::weight_cache::GpuWeightHandle,
        q_rows: usize,
        k_rows: usize,
        h: usize,
    ) -> Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>, GpuError> {
        match self
            .0
            .batch_attn_phase(hidden, norm_weight, norm_eps, qkv_handle, q_rows, k_rows, h)
        {
            Ok(result) => Ok(Some(result)),
            Err(e) => {
                tracing::warn!(error = %e, "batch_attn_phase failed, falling back");
                Ok(None)
            }
        }
    }

    fn batch_ffn_phase(
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
    ) -> Result<bool, GpuError> {
        match self.0.batch_ffn_phase(
            hidden,
            attn_out,
            norm_weight,
            norm_eps,
            attn_proj_handle,
            gate_up_handle,
            down_handle,
            h,
            intermediate,
            attn_proj_k,
        ) {
            Ok(()) => Ok(true),
            Err(e) => {
                tracing::warn!(error = %e, "batch_ffn_phase failed, falling back");
                Ok(false)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// select_backend
// ═══════════════════════════════════════════════════════════════════════════

/// Select the best available backend automatically.
///
/// Priority order (highest to lowest):
/// 1. [`Scirs2Backend`] (feature = "gpu") — Metal-accelerated via scirs2-core
/// 2. `NativeCudaBackend` (feature = "native-cuda") — direct cudarc CUDA
/// 3. CUDA stub (feature = "cuda", no "native-cuda") — falls back to CPU
/// 4. Metal stub (feature = "metal", macOS only) — falls back to CPU
/// 5. [`CpuBackend`] (always available)
///
/// If initialisation fails at any level the function falls through to the
/// next option, ultimately always returning a functional `CpuBackend`.
pub fn select_backend() -> Box<dyn GpuBackendTrait> {
    // `select_backend` may be called several times in a process (model load,
    // engine init, tests). The "scirs2 not accelerated" / "init failed" warnings
    // are properties of the host environment, not of any individual call site,
    // so emit each variant at most once per process.
    #[cfg(feature = "gpu")]
    use std::sync::atomic::{AtomicBool, Ordering};
    #[cfg(feature = "gpu")]
    fn warn_once(flag: &AtomicBool, msg: impl FnOnce()) {
        if !flag.swap(true, Ordering::Relaxed) {
            msg();
        }
    }

    // ── 1. Try scirs2-core GPU backend (Metal on macOS) ─────────────────
    #[cfg(feature = "gpu")]
    {
        static SCIRS2_NOT_ACCEL: AtomicBool = AtomicBool::new(false);
        static SCIRS2_INIT_FAIL: AtomicBool = AtomicBool::new(false);
        match Scirs2Backend::global() {
            Ok(b) => {
                if b.is_accelerated() {
                    return Box::new(Scirs2BackendHandle(b));
                }
                // scirs2-core returned a CPU context; skip and try stubs.
                warn_once(&SCIRS2_NOT_ACCEL, || {
                    warn!(
                        "select_backend: Scirs2Backend is not accelerated (backend={}), trying next",
                        b.backend_name()
                    );
                });
            }
            Err(e) => {
                warn_once(&SCIRS2_INIT_FAIL, || {
                    warn!("select_backend: Scirs2Backend init failed ({e}), trying next");
                });
            }
        }
    }

    // ── 2. Native CUDA backend (direct cudarc, Linux/Windows) ───────────
    #[cfg(all(
        feature = "native-cuda",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        match NativeCudaBackend::new() {
            Ok(b) => {
                tracing::info!("select_backend: NativeCudaBackend initialised");
                return Box::new(b);
            }
            Err(e) => {
                warn!("select_backend: NativeCudaBackend init failed ({e}), trying next");
            }
        }
    }

    // ── 3. CUDA stub ─────────────────────────────────────────────────────
    #[cfg(feature = "cuda")]
    {
        match CudaBackend::new() {
            Ok(b) => {
                return Box::new(b);
            }
            Err(e) => {
                warn!("select_backend: CUDA init failed ({e}), trying next");
            }
        }
    }

    // ── 3. Metal stub ───────────────────────────────────────────────────
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match MetalBackend::new() {
            Ok(b) => {
                return Box::new(b);
            }
            Err(e) => {
                warn!("select_backend: Metal init failed ({e}), trying CPU");
            }
        }
    }

    // ── 4. CPU fallback ─────────────────────────────────────────────────
    Box::new(CpuBackend::new())
}

// ═══════════════════════════════════════════════════════════════════════════
// gpu_matmul utility
// ═══════════════════════════════════════════════════════════════════════════

/// Perform a general matrix multiplication **C = A · B** using a GPU backend.
///
/// - `a` — row-major `[m, k]` matrix (length `m * k`)
/// - `b` — row-major `[k, n]` matrix (length `k * n`)
/// - Returns a row-major `[m, n]` matrix (length `m * n`)
///
/// This is implemented as `n` calls to `backend.matvec` (one per column of B)
/// and is provided as a convenience for callers that do not wish to manage
/// `DeviceBuffer` objects directly.
pub fn gpu_matmul(
    backend: &dyn GpuBackendTrait,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    device_id: usize,
) -> Result<Vec<f32>, GpuError> {
    if a.len() != m * k {
        return Err(GpuError::InvalidArgument(format!(
            "a.len()={} does not match m={} k={}",
            a.len(),
            m,
            k
        )));
    }
    if b.len() != k * n {
        return Err(GpuError::InvalidArgument(format!(
            "b.len()={} does not match k={} n={}",
            b.len(),
            k,
            n
        )));
    }

    let a_buf = backend.host_to_device(a, device_id)?;

    let mut c = vec![0.0_f32; m * n];

    for col in 0..n {
        let b_col: Vec<f32> = (0..k).map(|row| b[row * n + col]).collect();
        let x_buf = backend.host_to_device(&b_col, device_id)?;
        let y_buf = backend.matvec(&a_buf, &x_buf, m, k, device_id)?;
        let y = backend.device_to_host(&y_buf)?;

        for row in 0..m {
            c[row * n + col] = y[row];
        }
    }

    backend.synchronize(device_id)?;
    Ok(c)
}

// ═══════════════════════════════════════════════════════════════════════════
// gpu_gemv_1bit — high-level Q1_0_g128 GPU GEMV
// ═══════════════════════════════════════════════════════════════════════════

/// Perform Q1_0_g128 matrix-vector multiply on the GPU.
///
/// This is the primary entry point for callers that have raw
/// `BlockQ1_0G128` data and want GPU-accelerated inference.
///
/// # Arguments
/// - `block_bytes` — `&[u8]` raw bytes of `BlockQ1_0G128[]` (18 bytes each)
/// - `input` — `&[f32]` of length `k`
/// - `n_rows` — number of weight matrix rows
/// - `k` — input dimension (must be a multiple of 128)
///
/// # Returns
/// `Vec<f32>` of length `n_rows`, or falls back to CPU dequant+GEMV if GPU
/// is not available.
///
/// # Feature gates
/// Requires the `gpu` feature.  Without it, this function is still available
/// but always uses the CPU fallback path.
pub fn gpu_gemv_1bit(
    block_bytes: &[u8],
    input: &[f32],
    n_rows: usize,
    k: usize,
) -> Result<Vec<f32>, GpuError> {
    #[cfg(feature = "gpu")]
    {
        match Scirs2Backend::global() {
            Ok(backend) => {
                if backend.is_accelerated() {
                    return backend.gemv_q1_g128(block_bytes, input, n_rows, k);
                }
                // Non-accelerated (CPU) scirs2 context — use our own CPU path.
            }
            Err(e) => {
                warn!("gpu_gemv_1bit: GPU init failed ({e}), using CPU fallback");
            }
        }
    }

    // CPU fallback: dequant + scalar GEMV.
    cpu_gemv_1bit_fallback(block_bytes, input, n_rows, k)
}

/// CPU fallback for Q1_0_g128 GEMV.
///
/// Dequantises blocks inline and computes the dot-product per row.
fn cpu_gemv_1bit_fallback(
    block_bytes: &[u8],
    input: &[f32],
    n_rows: usize,
    k: usize,
) -> Result<Vec<f32>, GpuError> {
    if k == 0 || k % 128 != 0 {
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
    let blocks_per_row = k / 128;
    let block_size = 18_usize;
    let expected = n_rows * blocks_per_row * block_size;
    if block_bytes.len() < expected {
        return Err(GpuError::InvalidArgument(format!(
            "block_bytes too small: {} < {}",
            block_bytes.len(),
            expected,
        )));
    }

    let mut output = vec![0.0_f32; n_rows];

    for (row, output_val) in output.iter_mut().enumerate().take(n_rows) {
        let mut sum = 0.0_f32;
        for b in 0..blocks_per_row {
            let block_idx = row * blocks_per_row + b;
            let off = block_idx * block_size;

            // Read FP16 scale factor (little-endian).
            let d_bits = u16::from_le_bytes([block_bytes[off], block_bytes[off + 1]]);
            let scale = half::f16::from_bits(d_bits).to_f32();

            let input_base = b * 128;
            // Process 4 × u32 = 128 bits.
            for w in 0..4_usize {
                let byte_off = off + 2 + w * 4;
                let bits = u32::from_le_bytes([
                    block_bytes[byte_off],
                    block_bytes[byte_off + 1],
                    block_bytes[byte_off + 2],
                    block_bytes[byte_off + 3],
                ]);
                let base = input_base + w * 32;
                for i in 0..32_usize {
                    let sign = if (bits >> i) & 1 == 1 {
                        1.0_f32
                    } else {
                        -1.0_f32
                    };
                    sum += scale * sign * input[base + i];
                }
            }
        }
        *output_val = sum;
    }

    Ok(output)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_buffer_new_zeroed() {
        let buf = DeviceBuffer::new(4, 0);
        assert_eq!(buf.size(), 4);
        assert_eq!(buf.device_id(), 0);
        assert!(buf.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn device_buffer_from_slice_roundtrip() {
        let src = [1.0_f32, 2.0, 3.0];
        let buf = DeviceBuffer::from_slice(&src, 1);
        assert_eq!(buf.to_vec(), src);
    }

    #[test]
    fn launch_config_for_zero_elements() {
        let cfg = LaunchConfig::for_n_elements(0);
        assert_eq!(cfg.grid_dim.0, 1);
    }

    #[test]
    fn cpu_softmax_empty() {
        let backend = CpuBackend::new();
        let buf = DeviceBuffer::new(0, 0);
        let out = backend.softmax(&buf, 0, 0).expect("softmax empty");
        assert_eq!(out.size(), 0);
    }

    // ── CPU fallback GEMV tests ─────────────────────────────────────────

    #[test]
    fn cpu_gemv_1bit_identity_scale() {
        // 1 row, k=128, all bits set (weight = +1), scale = 1.0
        let scale = half::f16::from_f32(1.0);
        let scale_bytes = scale.to_bits().to_le_bytes();

        let mut block = vec![0u8; 18];
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        // Set all 128 bits to 1 → all weights = +scale = +1
        block[2..18].fill(0xFF);

        let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let expected: f32 = input.iter().sum(); // sum(0..128) = 8128

        let result =
            cpu_gemv_1bit_fallback(&block, &input, 1, 128).expect("cpu_gemv_1bit_fallback");
        assert!(
            (result[0] - expected).abs() < 1e-2,
            "got {} expected {}",
            result[0],
            expected,
        );
    }

    #[test]
    fn cpu_gemv_1bit_negative_scale() {
        // All bits 0 → weight = -scale.  With scale=1.0 and input=1.0:
        // output = -1 * 128 * 1.0 = -128
        let scale = half::f16::from_f32(1.0);
        let scale_bytes = scale.to_bits().to_le_bytes();

        let mut block = vec![0u8; 18];
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        // qs all zero → weight = -scale

        let input = vec![1.0_f32; 128];
        let result =
            cpu_gemv_1bit_fallback(&block, &input, 1, 128).expect("cpu_gemv_1bit_fallback");
        assert!(
            (result[0] - (-128.0)).abs() < 1e-2,
            "got {} expected -128",
            result[0],
        );
    }

    #[test]
    fn cpu_gemv_1bit_bad_k() {
        let result = cpu_gemv_1bit_fallback(&[], &[], 0, 64);
        assert!(result.is_err());
    }

    #[test]
    fn gpu_gemv_1bit_without_gpu() {
        // gpu_gemv_1bit should fall back to CPU.
        let scale = half::f16::from_f32(1.0);
        let scale_bytes = scale.to_bits().to_le_bytes();

        let mut block = vec![0u8; 18];
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        block[2..18].fill(0xFF);

        let input: Vec<f32> = vec![1.0_f32; 128];
        let result = gpu_gemv_1bit(&block, &input, 1, 128).expect("gpu_gemv_1bit");
        assert!((result[0] - 128.0).abs() < 1e-2, "got {}", result[0]);
    }
}
