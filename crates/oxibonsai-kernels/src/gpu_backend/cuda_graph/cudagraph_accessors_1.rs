//! # CudaGraph - accessors Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::driver::CudaSlice;
use std::sync::Arc;

use crate::gpu_backend::cuda_graph_slot::UNATTRIBUTED_CUDA_MODEL_EPOCH;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Upload `f32` weights and cache them under `key`, **unattributed**.
    ///
    /// On the first call for `key`, the slice is copied to a device buffer and
    /// stored in `f32_weight_cache`.  Subsequent calls clone the cached `Arc`.
    ///
    /// Unlike [`Self::get_or_upload_weight_soa`], no SoA reformatting is performed;
    /// the data is uploaded verbatim as typed `f32` device memory.
    ///
    /// The upload is not attributed to any model epoch, so it is never freed by
    /// [`Self::release_model_epoch`] (finding F-M3) — callers that know which
    /// model they are loading should use
    /// [`Self::get_or_upload_f32_weight_for_epoch`] instead.
    pub fn get_or_upload_f32_weight(
        &self,
        key: u64,
        data: &[f32],
    ) -> Result<Arc<CudaSlice<f32>>, CudaGraphError> {
        self.get_or_upload_f32_weight_for_epoch(key, data, UNATTRIBUTED_CUDA_MODEL_EPOCH)
    }

    /// [`Self::get_or_upload_f32_weight`], attributing the upload to
    /// `model_epoch` so the model's `Drop` can free it (finding **F-M3**).
    pub fn get_or_upload_f32_weight_for_epoch(
        &self,
        key: u64,
        data: &[f32],
        model_epoch: u64,
    ) -> Result<Arc<CudaSlice<f32>>, CudaGraphError> {
        let cached = {
            let cache = self
                .f32_weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache.get(&key).map(Arc::clone)
        };
        if let Some(existing) = cached {
            // Register on a hit too: the first upload may have been
            // unattributed, and registration is idempotent.
            self.register_model_weight(model_epoch, key)?;
            return Ok(existing);
        }
        let d_buf = self
            .stream
            .clone_htod(data)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod f32: {e}")))?;
        let arc = Arc::new(d_buf);
        {
            let mut cache = self
                .f32_weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache.insert(key, Arc::clone(&arc));
        }
        // Registered outside the cache lock so the lock order is always
        // "weight cache, then epoch registry", never the reverse.
        self.register_model_weight(model_epoch, key)?;
        Ok(arc)
    }

    /// Evict a previously-uploaded `f32` weight from the cache.
    ///
    /// Dropping the cached [`Arc`] frees the device buffer once no other handle
    /// is outstanding. Intended for callers whose host weight buffers are
    /// **transient** — e.g. the dequantise-on-demand text encoder, which
    /// allocates a fresh f32 buffer per Linear so its base pointer (the cache
    /// `key`) is recycled across calls and is therefore unsafe as a long-lived
    /// identity. Evicting right after the GEMM forces the next
    /// [`get_or_upload_f32_weight`](Self::get_or_upload_f32_weight) to re-upload
    /// fresh data instead of returning a stale buffer that merely shares a
    /// recycled address. A `key` that is not present is a no-op.
    pub fn evict_f32_weight(&self, key: u64) -> Result<(), CudaGraphError> {
        let mut cache = self
            .f32_weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        cache.remove(&key);
        Ok(())
    }
}
