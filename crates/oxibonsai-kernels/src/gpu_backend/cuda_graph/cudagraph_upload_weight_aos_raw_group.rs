//! # CudaGraph - upload_weight_aos_raw_group Methods
//!
//! Upload raw AoS weight bytes to GPU without any reformatting.
//! Used for Q4_0 and Q8_0 weights which are already in the correct AoS layout.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::CudaSlice;
use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Upload raw AoS bytes to GPU without any reformatting.
    ///
    /// Used for Q4_0 and Q8_0 weights which are already in the correct AoS layout
    /// (unlike Q1/TQ2 which require AoS → SoA reformatting before use).
    ///
    /// On first call for `handle_id`: uploads `aos_bytes` as-is to GPU device memory
    /// and caches the `Arc<CudaSlice<u8>>`.  On subsequent calls for the same
    /// `handle_id`: returns the cached `Arc` immediately without re-uploading.
    pub fn get_or_upload_weight_aos_raw(
        &self,
        handle_id: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<CudaSlice<u8>>, CudaGraphError> {
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(existing) = cache.get(&handle_id) {
            return Ok(Arc::clone(existing));
        }
        let d_weight = self
            .stream
            .clone_htod(aos_bytes)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod raw AoS: {e}")))?;
        let arc = Arc::new(d_weight);
        cache.insert(handle_id, Arc::clone(&arc));
        Ok(arc)
    }

    /// Upload raw AoS bytes produced by a lazy initialiser closure, if not already cached.
    ///
    /// Identical to `get_or_upload_weight_aos_raw` but the byte data is produced
    /// lazily (only on the first call for `handle_id`), avoiding allocations on
    /// cache hits.  Useful for fused gate+up weight construction.
    pub fn get_or_upload_weight_aos_raw_lazy<F>(
        &self,
        handle_id: u64,
        make_bytes: F,
    ) -> Result<Arc<CudaSlice<u8>>, CudaGraphError>
    where
        F: FnOnce() -> Vec<u8>,
    {
        {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            if let Some(existing) = cache.get(&handle_id) {
                return Ok(Arc::clone(existing));
            }
        }
        // Cache miss — produce bytes and upload outside the lock.
        let bytes = make_bytes();
        let d_weight = self
            .stream
            .clone_htod(&bytes)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod lazy AoS: {e}")))?;
        let arc = Arc::new(d_weight);
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        cache.insert(handle_id, Arc::clone(&arc));
        Ok(arc)
    }
}

#[cfg(test)]
mod tests {
    /// The method exists (compile-time check only; runtime needs a CUDA device).
    #[test]
    fn test_upload_weight_aos_raw_exists() {
        // This test simply verifies the code compiles.  It does not require a GPU.
        // The actual upload path is exercised by the GPU-gated tests in
        // cuda_q_std_prefill.
    }
}
