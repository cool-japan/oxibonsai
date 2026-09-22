//! # NativeCudaBackend - Trait Implementations
//!
//! This module contains trait implementations for `NativeCudaBackend`.
//!
//! ## Implemented Traits
//!
//! - `GpuBackendTrait`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::{DeviceBuffer, GpuBackendTrait, GpuError};
use cudarc::driver::CudaSlice;
use std::sync::Arc;
use tracing::warn;

use super::cudagraph_type::CudaGraph;
use super::functions::alloc_handle_id;
use super::types::NativeCudaBackend;

impl GpuBackendTrait for NativeCudaBackend {
    fn name(&self) -> &'static str {
        "native-cuda"
    }
    fn is_accelerated(&self) -> bool {
        true
    }
    /// Real device count from the driver (finding **F-M4**).
    ///
    /// This was a hardcoded `1` — and it is the *only* `device_count` any caller
    /// can reach through `GpuBackendTrait`, so `CudaGraph::device_count` (added
    /// by CUDA-SAFETY) had no consumer at all and a multi-GPU host still
    /// reported one device.
    ///
    /// Floored at `1`: this backend only exists once `CudaGraph::global()` has
    /// built a context, which proves a device is present, so a driver query that
    /// fails here must not make the backend advertise zero devices while it is
    /// actively serving one.
    fn device_count(&self) -> usize {
        CudaGraph::device_count().max(1)
    }
    fn alloc(&self, size: usize, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        self.cpu_fallback.alloc(size, device_id)
    }
    fn host_to_device(&self, src: &[f32], device_id: usize) -> Result<DeviceBuffer, GpuError> {
        self.cpu_fallback.host_to_device(src, device_id)
    }
    fn device_to_host(&self, buf: &DeviceBuffer) -> Result<Vec<f32>, GpuError> {
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
        self.cpu_fallback.matvec(a, x, m, k, device_id)
    }
    fn relu(&self, x: &DeviceBuffer, device_id: usize) -> Result<DeviceBuffer, GpuError> {
        self.cpu_fallback.relu(x, device_id)
    }
    fn softmax(
        &self,
        x: &DeviceBuffer,
        size: usize,
        device_id: usize,
    ) -> Result<DeviceBuffer, GpuError> {
        self.cpu_fallback.softmax(x, size, device_id)
    }
    fn synchronize(&self, _device_id: usize) -> Result<(), GpuError> {
        self.graph
            .stream
            .synchronize()
            .map_err(|e| GpuError::SyncFailed(e.to_string()))
    }
    fn memory_info(&self, _device_id: usize) -> Result<(usize, usize), GpuError> {
        cudarc::driver::result::mem_get_info().map_err(|e| GpuError::NotAvailable(e.to_string()))
    }
    fn gemv_q1_g128(
        &self,
        block_bytes: &[u8],
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let handle_id = block_bytes.as_ptr() as u64;
        self.graph
            .encode_gemv(handle_id, block_bytes, input, n_rows, k)
            .map_err(|e| GpuError::KernelLaunch(e.to_string()))
    }
    fn upload_weights_raw(
        &self,
        block_bytes: &[u8],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        let id = alloc_handle_id();
        self.graph
            .upload_weight_soa_new(id, block_bytes)
            .map_err(|e| GpuError::KernelLaunch(e.to_string()))?;
        Ok(crate::weight_cache::GpuWeightHandle(id))
    }
    fn gemv_q1_g128_cached(
        &self,
        handle: crate::weight_cache::GpuWeightHandle,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.graph
            .encode_gemv_cached(handle.id(), input, n_rows, k)
            .map_err(|e| GpuError::KernelLaunch(e.to_string()))
    }
    fn upload_weights_ternary(
        &self,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    ) -> Result<crate::weight_cache::GpuWeightHandle, GpuError> {
        let id = alloc_handle_id();
        self.graph
            .upload_weight_tq2_soa(id, blocks)
            .map_err(|e| GpuError::KernelLaunch(e.to_string()))?;
        Ok(crate::weight_cache::GpuWeightHandle(id))
    }
    fn gemv_tq2_g128_cached(
        &self,
        handle: crate::weight_cache::GpuWeightHandle,
        input: &[f32],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.graph
            .encode_gemv_tq2_cached(handle.id(), input, n_rows, k)
            .map_err(|e| GpuError::KernelLaunch(e.to_string()))
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
        _attn_proj_k: usize,
    ) -> Result<bool, GpuError> {
        let lookup = |id: u64| -> Result<Arc<CudaSlice<u8>>, GpuError> {
            self.graph
                .weight_cache
                .lock()
                .map_err(|_| GpuError::SyncFailed("weight cache lock poisoned".into()))?
                .get(&id)
                .map(Arc::clone)
                .ok_or_else(|| GpuError::NotAvailable(format!("weight {id} not cached")))
        };
        let attn_proj_w = match lookup(attn_proj_handle.id()) {
            Ok(w) => w,
            Err(e) => {
                warn!(
                    error = % e,
                    "NativeCudaBackend::batch_ffn_phase: missing attn_proj weight"
                );
                return Ok(false);
            }
        };
        let gate_up_w = match lookup(gate_up_handle.id()) {
            Ok(w) => w,
            Err(e) => {
                warn!(
                    error = % e,
                    "NativeCudaBackend::batch_ffn_phase: missing gate_up weight"
                );
                return Ok(false);
            }
        };
        let down_w = match lookup(down_handle.id()) {
            Ok(w) => w,
            Err(e) => {
                warn!(
                    error = % e,
                    "NativeCudaBackend::batch_ffn_phase: missing down weight"
                );
                return Ok(false);
            }
        };
        self.graph
            .encode_ffn_phase(
                hidden,
                attn_out,
                norm_weight,
                norm_eps,
                &attn_proj_w,
                &gate_up_w,
                &down_w,
                h,
                intermediate,
            )
            .map_err(|e| GpuError::KernelLaunch(e.to_string()))?;
        Ok(true)
    }
}
