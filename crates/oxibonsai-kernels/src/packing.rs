//! Aligned memory allocation, block packing, and prefetch utilities.
//!
//! Provides cache-friendly memory layout tools for 1-bit quantized
//! weight data, including:
//! - 64-byte aligned buffers for optimal SIMD load/store operations
//! - Block reordering for sequential cache-line access during GEMV
//! - Software prefetch wrappers for all supported architectures
//! - Working set size estimation for cache-aware scheduling

use oxibonsai_core::tensor::{BlockQ1_0G128, QK1_0_G128};

use crate::error::{KernelError, KernelResult};

/// Required alignment in bytes for SIMD-optimal memory access.
/// 64 bytes = one cache line on most modern CPUs, also the alignment
/// needed for AVX-512 aligned loads/stores.
const CACHE_LINE_BYTES: usize = 64;

/// Size of one [`BlockQ1_0G128`] in bytes (2 bytes f16 scale + 16 bytes packed bits).
const BLOCK_SIZE_BYTES: usize = 18;

// ─── AlignedBuffer ─────────────────────────────────────────────────────

/// A 64-byte aligned f32 buffer for SIMD-friendly memory access.
///
/// Standard `Vec<f32>` provides only 4-byte alignment. This struct
/// guarantees 64-byte alignment using over-allocation with offset
/// tracking, ensuring optimal performance for AVX-512 aligned loads.
///
/// # Design
/// We allocate extra capacity and find the aligned start within the
/// allocation. The buffer stores `f32` values at the aligned offset.
pub struct AlignedBuffer {
    /// Raw storage with extra room for alignment padding.
    storage: Vec<u8>,
    /// Byte offset into `storage` where aligned data begins.
    offset: usize,
    /// Number of f32 elements.
    len: usize,
}

impl AlignedBuffer {
    /// Create a new zero-initialized aligned buffer holding `size` f32 values.
    ///
    /// The buffer is aligned to 64 bytes (cache line boundary).
    pub fn new(size: usize) -> Self {
        let byte_len = size * 4;
        // Allocate extra bytes for alignment padding
        let total = byte_len + CACHE_LINE_BYTES;
        let storage = vec![0u8; total];

        // Find the aligned offset within the allocation
        let base_ptr = storage.as_ptr() as usize;
        let aligned_ptr = (base_ptr + CACHE_LINE_BYTES - 1) & !(CACHE_LINE_BYTES - 1);
        let offset = aligned_ptr - base_ptr;

        Self {
            storage,
            offset,
            len: size,
        }
    }

    /// Return an immutable slice of the aligned f32 data.
    pub fn as_slice(&self) -> &[f32] {
        let ptr = self.storage[self.offset..].as_ptr() as *const f32;
        // SAFETY: We ensured alignment and allocated enough bytes for `self.len` f32s.
        unsafe { core::slice::from_raw_parts(ptr, self.len) }
    }

    /// Return a mutable slice of the aligned f32 data.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        let ptr = self.storage[self.offset..].as_mut_ptr() as *mut f32;
        // SAFETY: We ensured alignment and allocated enough bytes for `self.len` f32s.
        unsafe { core::slice::from_raw_parts_mut(ptr, self.len) }
    }

    /// Number of f32 elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check that the buffer is actually aligned to 64 bytes.
    pub fn is_aligned(&self) -> bool {
        let ptr = self.storage[self.offset..].as_ptr() as usize;
        ptr % CACHE_LINE_BYTES == 0
    }

    /// Copy data from a slice into the aligned buffer.
    ///
    /// Copies up to `self.len` elements, ignoring any excess in `src`.
    pub fn copy_from_slice(&mut self, src: &[f32]) {
        let copy_len = self.len.min(src.len());
        let dst = self.as_mut_slice();
        dst[..copy_len].copy_from_slice(&src[..copy_len]);
    }
}

// ─── Block packing ─────────────────────────────────────────────────────

/// Reorder blocks for cache-friendly sequential row access during GEMV.
///
/// The standard layout stores all blocks for row 0, then row 1, etc.
/// This function reorders blocks into tiles so that consecutive rows
/// within a tile are adjacent in memory, improving spatial locality
/// when the GEMV loop processes tiles of rows.
///
/// **Tile layout:** For `tile_size` rows, blocks are arranged as:
/// ```text
/// [row_0_block_0, row_1_block_0, ..., row_{tile-1}_block_0,
///  row_0_block_1, row_1_block_1, ..., row_{tile-1}_block_1, ...]
/// ```
///
/// This means all rows' first block are adjacent, then all rows'
/// second block, etc. — matching the GEMV's block-sequential access
/// pattern within each tile.
pub fn pack_blocks_for_gemv(
    blocks: &[BlockQ1_0G128],
    n_rows: usize,
    k: usize,
) -> KernelResult<Vec<BlockQ1_0G128>> {
    if k % QK1_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    let blocks_per_row = k / QK1_0_G128;
    let total_blocks = n_rows * blocks_per_row;
    if blocks.len() < total_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: total_blocks,
            available: blocks.len(),
        });
    }

    let tile_size = 32_usize.min(n_rows);
    let mut packed = Vec::with_capacity(total_blocks);

    let mut tile_start = 0;
    while tile_start < n_rows {
        let tile_rows = (n_rows - tile_start).min(tile_size);

        // Interleave: for each block index, emit that block from all rows in the tile
        for bi in 0..blocks_per_row {
            for row_offset in 0..tile_rows {
                let row = tile_start + row_offset;
                let src_idx = row * blocks_per_row + bi;
                packed.push(blocks[src_idx]);
            }
        }

        tile_start += tile_rows;
    }

    Ok(packed)
}

/// Unpack blocks from tiled layout back to standard row-major layout.
///
/// Inverse of [`pack_blocks_for_gemv`]. Useful for verifying round-trip
/// correctness in tests.
pub fn unpack_blocks_from_gemv(
    packed: &[BlockQ1_0G128],
    n_rows: usize,
    k: usize,
) -> KernelResult<Vec<BlockQ1_0G128>> {
    if k % QK1_0_G128 != 0 {
        return Err(KernelError::NotBlockAligned {
            count: k,
            block_size: QK1_0_G128,
        });
    }
    let blocks_per_row = k / QK1_0_G128;
    let total_blocks = n_rows * blocks_per_row;
    if packed.len() < total_blocks {
        return Err(KernelError::BufferTooSmall {
            needed: total_blocks,
            available: packed.len(),
        });
    }

    let tile_size = 32_usize.min(n_rows);
    let mut unpacked = vec![
        BlockQ1_0G128 {
            d: half::f16::from_f32(0.0),
            qs: [0u8; 16],
        };
        total_blocks
    ];

    let mut packed_idx = 0;
    let mut tile_start = 0;
    while tile_start < n_rows {
        let tile_rows = (n_rows - tile_start).min(tile_size);

        for bi in 0..blocks_per_row {
            for row_offset in 0..tile_rows {
                let row = tile_start + row_offset;
                let dst_idx = row * blocks_per_row + bi;
                unpacked[dst_idx] = packed[packed_idx];
                packed_idx += 1;
            }
        }

        tile_start += tile_rows;
    }

    Ok(unpacked)
}

// ─── Prefetch hints ────────────────────────────────────────────────────

/// Emit a software prefetch hint for read access.
///
/// On AArch64: uses `__prefetch` intrinsic for data read into L1.
/// On x86_64: uses `_mm_prefetch` with `_MM_HINT_T0` (all cache levels).
/// On other targets: no-op (compiler may still generate prefetch if it sees fit).
///
/// This is a *hint* — the CPU is free to ignore it. The benefit comes
/// from overlapping prefetch latency with computation in inner loops.
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: prefetch is always safe, it's just a hint. The
        // `aarch64_prefetch!` macro supplies the `unsafe` block and degrades to
        // a no-op off-nightly (where the intrinsic is unavailable).
        crate::aarch64_prefetch!(ptr as *const i8, 0, 3);
    }

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: prefetch is always safe, it's just a hint
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }

    // No-op fallback for other architectures
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

/// Emit a software prefetch hint for write access.
///
/// Similar to [`prefetch_read`] but hints that the cache line will
/// be written to, potentially triggering an exclusive prefetch.
#[inline(always)]
pub fn prefetch_write<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: prefetch is always safe. The `aarch64_prefetch!` macro supplies
        // the `unsafe` block and degrades to a no-op off-nightly.
        // _prefetch with pst=1 means prefetch for store.
        crate::aarch64_prefetch!(ptr as *const i8, 1, 3);
    }

    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: prefetch is always safe
        unsafe {
            // _MM_HINT_ET0: exclusive prefetch to all cache levels
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

// ─── Working set estimation ────────────────────────────────────────────

/// Estimate the working set size in bytes for a GEMV/GEMM computation.
///
/// Accounts for:
/// - Weight blocks: `n_rows * blocks_per_row * 18` bytes
/// - Input vector: `k * 4` bytes
/// - Output vector: `n_rows * 4` bytes
///
/// Use this to decide whether tiling or streaming strategies are needed.
pub fn estimate_working_set_bytes(n_rows: usize, k: usize) -> usize {
    let blocks_per_row = k / QK1_0_G128;
    let weight_bytes = n_rows * blocks_per_row * BLOCK_SIZE_BYTES;
    let input_bytes = k * core::mem::size_of::<f32>();
    let output_bytes = n_rows * core::mem::size_of::<f32>();
    weight_bytes + input_bytes + output_bytes
}

/// Check if the working set fits within L1 cache (~32 KB).
pub fn fits_in_l1(n_rows: usize, k: usize) -> bool {
    estimate_working_set_bytes(n_rows, k) <= 32 * 1024
}

/// Check if the working set fits within L2 cache (~256 KB).
pub fn fits_in_l2(n_rows: usize, k: usize) -> bool {
    estimate_working_set_bytes(n_rows, k) <= 256 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
        BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: bits,
        }
    }

    #[test]
    fn aligned_buffer_creation() {
        let buf = AlignedBuffer::new(256);
        assert_eq!(buf.len(), 256);
        assert!(!buf.is_empty());
        assert!(buf.is_aligned(), "buffer should be 64-byte aligned");
    }

    #[test]
    fn aligned_buffer_zero_initialized() {
        let buf = AlignedBuffer::new(128);
        for &v in buf.as_slice() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn aligned_buffer_read_write() {
        let mut buf = AlignedBuffer::new(4);
        {
            let slice = buf.as_mut_slice();
            slice[0] = 1.0;
            slice[1] = 2.0;
            slice[2] = 3.0;
            slice[3] = 4.0;
        }
        let slice = buf.as_slice();
        assert!((slice[0] - 1.0).abs() < f32::EPSILON);
        assert!((slice[1] - 2.0).abs() < f32::EPSILON);
        assert!((slice[2] - 3.0).abs() < f32::EPSILON);
        assert!((slice[3] - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn aligned_buffer_copy_from_slice() {
        let mut buf = AlignedBuffer::new(4);
        let src = [10.0f32, 20.0, 30.0, 40.0];
        buf.copy_from_slice(&src);
        assert!((buf.as_slice()[2] - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn aligned_buffer_empty() {
        let buf = AlignedBuffer::new(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let n_rows = 10;
        let k = 256;
        let blocks_per_row = k / QK1_0_G128;
        let mut blocks = Vec::new();
        for row in 0..n_rows {
            for bi in 0..blocks_per_row {
                let bits = [((row * 37 + bi * 13) & 0xFF) as u8; 16];
                blocks.push(make_block(0.5 + row as f32 * 0.1, bits));
            }
        }

        let packed = pack_blocks_for_gemv(&blocks, n_rows, k).expect("packing should succeed");
        let unpacked =
            unpack_blocks_from_gemv(&packed, n_rows, k).expect("unpacking should succeed");

        assert_eq!(blocks.len(), unpacked.len());
        for (i, (orig, restored)) in blocks.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(
                orig.d.to_f32(),
                restored.d.to_f32(),
                "scale mismatch at block {i}"
            );
            assert_eq!(orig.qs, restored.qs, "bits mismatch at block {i}");
        }
    }

    #[test]
    fn pack_single_row() {
        let blocks = vec![make_block(1.0, [0xAA; 16])];
        let packed = pack_blocks_for_gemv(&blocks, 1, 128).expect("pack should succeed");
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].qs, [0xAA; 16]);
    }

    #[test]
    fn pack_validation_errors() {
        let blocks = vec![make_block(1.0, [0xFF; 16])];

        // Not block aligned
        let result = pack_blocks_for_gemv(&blocks, 1, 100);
        assert!(result.is_err());

        // Too few blocks
        let result = pack_blocks_for_gemv(&blocks, 2, 128);
        assert!(result.is_err());
    }

    #[test]
    fn working_set_estimation() {
        let ws = estimate_working_set_bytes(32, 128);
        // 32 rows * 1 block/row * 18 bytes + 128 * 4 (input) + 32 * 4 (output)
        let expected = 32 * 18 + 128 * 4 + 32 * 4;
        assert_eq!(ws, expected);
    }

    #[test]
    fn fits_in_l1_small() {
        // Small problem should fit
        assert!(fits_in_l1(4, 128));
    }

    #[test]
    fn fits_in_l1_large() {
        // Large problem should not fit
        assert!(!fits_in_l1(1000, 4096));
    }

    #[test]
    fn fits_in_l2_moderate() {
        // Moderate problem should fit in L2
        assert!(fits_in_l2(64, 256));
    }

    #[test]
    fn prefetch_does_not_crash() {
        // Just verify prefetch hints don't cause issues
        let data = vec![1.0f32; 64];
        prefetch_read(data.as_ptr());
        prefetch_write(data.as_ptr());
    }
}
