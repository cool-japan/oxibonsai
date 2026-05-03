//! # CudaGraph - reformat_tq2_blocks_to_soa_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Reformat TQ2_0_g128 blocks (`qs: [u8;32]`, `d: f16`) to SoA bytes:
    /// `[N×2 bytes FP16 scales LE][N×32 bytes qs]`.
    fn reformat_tq2_blocks_to_soa(blocks: &[oxibonsai_core::BlockTQ2_0_g128]) -> Vec<u8> {
        let n = blocks.len();
        let mut soa = Vec::with_capacity(n * 34);
        for block in blocks {
            let bits = block.d.to_bits().to_le_bytes();
            soa.push(bits[0]);
            soa.push(bits[1]);
        }
        for block in blocks {
            soa.extend_from_slice(&block.qs);
        }
        soa
    }
    /// Upload TQ2_0_g128 weights in SoA layout under a new handle id.
    pub fn upload_weight_tq2_soa(
        &self,
        handle_id: u64,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    ) -> Result<(), CudaGraphError> {
        let soa = Self::reformat_tq2_blocks_to_soa(blocks);
        let d_weight = self
            .stream
            .clone_htod(&soa)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod tq2: {e}")))?;
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        cache.insert(handle_id, Arc::new(d_weight));
        Ok(())
    }

    /// Reformat raw TQ2_0_g128 AoS bytes to SoA layout.
    ///
    /// Each AoS block is 34 bytes: `[d: f16 LE (2 bytes)][qs: [u8; 32]]`.
    /// SoA output is `[N×2 bytes FP16 scales][N×32 bytes qs]`.
    ///
    /// Returns `None` when `aos_bytes.len()` is not a multiple of 34.
    fn reformat_tq2_aos_bytes_to_soa(aos_bytes: &[u8]) -> Option<Vec<u8>> {
        const BLOCK_BYTES: usize = 34;
        const SCALE_BYTES: usize = 2;
        const QS_BYTES: usize = 32;
        if aos_bytes.is_empty() || aos_bytes.len() % BLOCK_BYTES != 0 {
            return None;
        }
        let n = aos_bytes.len() / BLOCK_BYTES;
        let mut soa = Vec::with_capacity(aos_bytes.len());
        // Scales pass
        for i in 0..n {
            let src = i * BLOCK_BYTES;
            soa.extend_from_slice(&aos_bytes[src..src + SCALE_BYTES]);
        }
        // Quant codes pass
        for i in 0..n {
            let src = i * BLOCK_BYTES + SCALE_BYTES;
            soa.extend_from_slice(&aos_bytes[src..src + QS_BYTES]);
        }
        Some(soa)
    }

    /// Return a cached TQ2 weight slice, or reformat AoS bytes to SoA and upload.
    ///
    /// Mirrors `get_or_upload_weight_soa` but for TQ2_0_g128 block layout
    /// (34 bytes/block: 2-byte FP16 scale + 32-byte qs) rather than Q1_0_g128
    /// (18 bytes/block: 2-byte FP16 scale + 16-byte data).
    pub fn get_or_upload_weight_tq2_soa(
        &self,
        handle_id: u64,
        aos_bytes: &[u8],
    ) -> Result<Arc<cudarc::driver::CudaSlice<u8>>, CudaGraphError> {
        {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            if let Some(existing) = cache.get(&handle_id) {
                return Ok(Arc::clone(existing));
            }
        }
        let soa = Self::reformat_tq2_aos_bytes_to_soa(aos_bytes).ok_or_else(|| {
            CudaGraphError::WeightLayoutError(format!(
                "TQ2 AoS bytes length {} not divisible by 34",
                aos_bytes.len()
            ))
        })?;
        let d_weight = self
            .stream
            .clone_htod(&soa)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod tq2_soa: {e}")))?;
        let arc = Arc::new(d_weight);
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        cache.insert(handle_id, Arc::clone(&arc));
        Ok(arc)
    }

    /// Return a cached TQ2 weight slice, using a lazy producer when not yet cached.
    ///
    /// On cache miss, calls `make_bytes()` to build the AoS byte slice, then reformats
    /// to SoA and uploads.  Avoids building the bytes when the weight is already cached.
    pub fn get_or_upload_weight_tq2_soa_lazy<F>(
        &self,
        handle_id: u64,
        make_bytes: F,
    ) -> Result<Arc<cudarc::driver::CudaSlice<u8>>, CudaGraphError>
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
        let aos_bytes = make_bytes();
        self.get_or_upload_weight_tq2_soa(handle_id, &aos_bytes)
    }
}
