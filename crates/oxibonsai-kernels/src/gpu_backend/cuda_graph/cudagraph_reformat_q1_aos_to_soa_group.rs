//! # CudaGraph - reformat_q1_aos_to_soa_group Methods
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
    /// Reformat Q1_0_G128 weight bytes from AoS to SoA layout.
    ///
    /// AoS: `[scale₀|data₀][scale₁|data₁]...` (18 bytes/block)
    /// SoA: `[scale₀…scaleₙ][data₀…dataₙ]`
    ///
    /// Returns `None` if `aos_bytes.len()` is not a multiple of 18.
    pub(crate) fn reformat_q1_aos_to_soa(aos_bytes: &[u8]) -> Option<Vec<u8>> {
        const BLOCK_BYTES: usize = 18;
        const SCALE_BYTES: usize = 2;
        const DATA_BYTES: usize = 16;
        if aos_bytes.is_empty() || !aos_bytes.len().is_multiple_of(BLOCK_BYTES) {
            return None;
        }
        let n_blocks = aos_bytes.len() / BLOCK_BYTES;
        let mut soa = vec![0u8; n_blocks * BLOCK_BYTES];
        let (scales_section, data_section) = soa.split_at_mut(n_blocks * SCALE_BYTES);
        for i in 0..n_blocks {
            let src = i * BLOCK_BYTES;
            scales_section[i * SCALE_BYTES..i * SCALE_BYTES + SCALE_BYTES]
                .copy_from_slice(&aos_bytes[src..src + SCALE_BYTES]);
            data_section[i * DATA_BYTES..i * DATA_BYTES + DATA_BYTES]
                .copy_from_slice(&aos_bytes[src + SCALE_BYTES..src + BLOCK_BYTES]);
        }
        Some(soa)
    }
    /// Return a cached weight slice or upload it on demand, **unattributed**.
    ///
    /// On first call for `handle_id`: converts `aos_bytes` to SoA, uploads to GPU,
    /// caches the slice.  On subsequent calls: returns the cached `Arc` immediately.
    ///
    /// The upload is not attributed to any model epoch, so it is never freed by
    /// [`Self::release_model_epoch`] (finding F-M3); use
    /// [`Self::get_or_upload_weight_soa_for_epoch`] when the owning model is
    /// known.
    pub fn get_or_upload_weight_soa(
        &self,
        handle_id: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<CudaSlice<u8>>, CudaGraphError> {
        self.get_or_upload_weight_soa_for_epoch(handle_id, aos_bytes, UNATTRIBUTED_CUDA_MODEL_EPOCH)
    }

    /// [`Self::get_or_upload_weight_soa`], attributing the upload to
    /// `model_epoch` so the model's `Drop` can free it (finding **F-M3**).
    pub fn get_or_upload_weight_soa_for_epoch(
        &self,
        handle_id: u64,
        aos_bytes: &[u8],
        model_epoch: u64,
    ) -> Result<Arc<CudaSlice<u8>>, CudaGraphError> {
        let cached = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache.get(&handle_id).map(Arc::clone)
        };
        if let Some(existing) = cached {
            self.register_model_weight(model_epoch, handle_id)?;
            return Ok(existing);
        }
        let soa = Self::reformat_q1_aos_to_soa(aos_bytes).ok_or_else(|| {
            CudaGraphError::WeightLayoutError(format!(
                "AoS bytes length {} not divisible by 18",
                aos_bytes.len()
            ))
        })?;
        let d_weight = self
            .stream
            .clone_htod(&soa)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod weight: {e}")))?;
        let arc = Arc::new(d_weight);
        {
            let mut cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache.insert(handle_id, Arc::clone(&arc));
        }
        // Outside the cache lock: the lock order is always "weight cache, then
        // epoch registry".
        self.register_model_weight(model_epoch, handle_id)?;
        Ok(arc)
    }

    /// [`Self::get_or_upload_weight_soa_lazy`], attributing the upload to
    /// `model_epoch` (finding **F-M3**).
    ///
    /// Lives here rather than beside the unattributed lazy variant because that
    /// file is outside this package's ownership; the byte producer is still only
    /// invoked on a cache miss.
    pub fn get_or_upload_weight_soa_lazy_for_epoch<F>(
        &self,
        handle_id: u64,
        make_bytes: F,
        model_epoch: u64,
    ) -> Result<Arc<CudaSlice<u8>>, CudaGraphError>
    where
        F: FnOnce() -> Vec<u8>,
    {
        let cached = {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            cache.get(&handle_id).map(Arc::clone)
        };
        if let Some(existing) = cached {
            self.register_model_weight(model_epoch, handle_id)?;
            return Ok(existing);
        }
        let aos_bytes = make_bytes();
        self.get_or_upload_weight_soa_for_epoch(handle_id, &aos_bytes, model_epoch)
    }
}
