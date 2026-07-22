//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::types::{
    CudaActivationBuffers, CudaModules, LmHeadBuffers, QkvBuffers, TernaryGemvBuffers,
};

/// Direct CUDA dispatch engine, mirroring `MetalGraph` for Linux/Windows.
///
/// Owns the CUDA context, stream, compiled kernels, weight cache, and
/// activation buffer pool. All state is protected by `Mutex` to satisfy
/// `Send + Sync` for `OnceLock<Arc<CudaGraph>>` storage.
pub struct CudaGraph {
    #[allow(dead_code)]
    pub(super) context: Arc<CudaContext>,
    pub(super) stream: Arc<CudaStream>,
    pub(super) modules: CudaModules,
    pub(super) buffers: Mutex<Option<CudaActivationBuffers>>,
    pub(super) qkv_buffers: Mutex<Option<QkvBuffers>>,
    pub(super) weight_cache: Mutex<HashMap<u64, Arc<CudaSlice<u8>>>>,
    /// Separate cache for f32 tensors (norm weights, RoPE buffers, etc.)
    pub(super) f32_weight_cache: Mutex<HashMap<u64, Arc<CudaSlice<f32>>>>,
    /// Pre-allocated GPU buffers for the LM-head GEMV.
    pub(super) lm_head_buffers: Mutex<Option<LmHeadBuffers>>,
    /// Reusable input/output buffers for `encode_gemv_tq2_cached`.
    pub(super) tq2_gemv_buffers: Mutex<Option<TernaryGemvBuffers>>,
}
