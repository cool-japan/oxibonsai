//! # CudaGraph - v8_shared_bytes_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::CudaSlice;
use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Shared-memory bytes required for the V8 kernel at given `k`.
    /// Returns `None` when `k` exceeds the 48 KB default shared-mem limit.
    #[inline]
    pub(crate) fn v8_shared_bytes(k: usize) -> Option<u32> {
        super::super::cuda_kernels::v8_shared_mem_bytes(k, 49_152)
    }
    /// Public auto-dispatch GEMV: uses V8 when `k` fits in shared mem, V7 otherwise.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    pub unsafe fn launch_gemv_pub(
        &self,
        d_weight: &Arc<CudaSlice<u8>>,
        d_input: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        match Self::v8_shared_bytes(k as usize) {
            Some(smem) => self.launch_gemv_v8(d_weight, d_input, d_output, n_rows, k, smem),
            None => self.launch_gemv_v9(d_weight, d_input, d_output, n_rows, k),
        }
    }
    /// Public auto-dispatch GEMV with fused in-place residual add.
    ///
    /// Computes `d_inout[row] = dot(weight[row], d_input) + d_inout[row]` for all rows.
    /// Uses V8 (shared-memory cache) when k fits in 49 KB; V9 (vectorised loads) otherwise.
    ///
    /// # Safety
    /// All slices must be valid device pointers on `self.stream`.
    /// `d_inout` is used as both the residual source and the output destination.
    /// Each output row is written exactly once by a single warp, so the read-before-write
    /// within the fused kernel is data-race-free even with aliased pointers.
    pub unsafe fn launch_gemv_residual_pub(
        &self,
        d_weight: &Arc<CudaSlice<u8>>,
        d_input: &CudaSlice<f32>,
        d_inout: &mut CudaSlice<f32>,
        n_rows: u32,
        k: u32,
    ) -> Result<(), CudaGraphError> {
        let d_residual = &*(d_inout as *const CudaSlice<f32>);
        match Self::v8_shared_bytes(k as usize) {
            Some(smem) => self
                .launch_gemv_v8_residual(d_weight, d_input, d_residual, d_inout, n_rows, k, smem),
            None => self.launch_gemv_v9_residual(d_weight, d_input, d_residual, d_inout, n_rows, k),
        }
    }
}
