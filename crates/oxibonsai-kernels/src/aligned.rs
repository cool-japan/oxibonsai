//! Cache-line aligned memory allocations for SIMD kernel operations.
//!
//! Standard `Vec<f32>` alignment (4 bytes) is insufficient for optimal SIMD
//! loads and stores which require 32-byte (AVX) or 64-byte (AVX-512/cache line)
//! alignment. This module provides aligned buffer types that guarantee 64-byte
//! alignment for all allocations.

use std::alloc::Layout;

use oxibonsai_core::tensor::BlockQ1_0G128;

/// Alignment in bytes for all aligned allocations (cache line size).
pub const ALIGNMENT: usize = 64;

/// A cache-line aligned buffer of `f32` values.
///
/// Guarantees that the backing memory starts at a 64-byte boundary,
/// which is optimal for AVX-512 aligned loads and cache line prefetch.
///
/// # Example
///
/// ```
/// use oxibonsai_kernels::aligned::AlignedBuffer;
///
/// let buf = AlignedBuffer::new(256);
/// assert_eq!(buf.len(), 256);
/// assert_eq!(buf.as_ptr() as usize % 64, 0);
/// ```
pub struct AlignedBuffer {
    /// Raw pointer to the aligned allocation.
    ptr: *mut f32,
    /// Number of f32 elements.
    len: usize,
    /// Layout used for deallocation.
    layout: Layout,
}

// SAFETY: The buffer owns its allocation exclusively.
unsafe impl Send for AlignedBuffer {}
// SAFETY: Shared references to the buffer are read-only.
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate a new zero-initialized aligned buffer of `len` f32 elements.
    ///
    /// The returned buffer is guaranteed to have 64-byte alignment.
    /// A zero-length buffer produces a valid (dangling but aligned) pointer.
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: ALIGNMENT as *mut f32, // aligned dangling pointer
                len: 0,
                layout: Layout::from_size_align(0, ALIGNMENT)
                    .expect("zero-size layout should always be valid"),
            };
        }

        // Guard against `usize` wraparound: an oversized `len` must not silently
        // wrap to a small `byte_size` (which would allocate too little memory while
        // `self.len` stays large, making `as_slice` produce an out-of-bounds slice).
        let byte_size = len
            .checked_mul(std::mem::size_of::<f32>())
            .expect("AlignedBuffer allocation size overflows usize");
        let layout = Layout::from_size_align(byte_size, ALIGNMENT)
            .expect("layout should be valid for reasonable buffer sizes");

        // SAFETY: layout has non-zero size and valid alignment.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        Self {
            ptr: ptr.cast::<f32>(),
            len,
            layout,
        }
    }

    /// Returns the number of f32 elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the raw pointer to the start of the buffer.
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    /// Returns a mutable raw pointer to the start of the buffer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    /// View the buffer as an immutable slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for len elements, properly aligned, and initialized to zero.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// View the buffer as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: ptr is valid for len elements, properly aligned, and we have exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Copy data from a slice into this buffer.
    ///
    /// Panics if `src.len() > self.len()`.
    pub fn copy_from_slice(&mut self, src: &[f32]) {
        assert!(
            src.len() <= self.len,
            "source slice length ({}) exceeds buffer length ({})",
            src.len(),
            self.len
        );
        self.as_mut_slice()[..src.len()].copy_from_slice(src);
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: ptr was allocated with this layout in `new`.
            unsafe {
                std::alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
            }
        }
    }
}

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("len", &self.len)
            .field("alignment", &ALIGNMENT)
            .field("aligned", &(self.as_ptr() as usize % ALIGNMENT == 0))
            .finish()
    }
}

/// A cache-line aligned buffer of `BlockQ1_0G128` values.
///
/// Same alignment guarantees as [`AlignedBuffer`] but for quantized weight blocks.
pub struct AlignedBlocks {
    /// Raw pointer to the aligned allocation.
    ptr: *mut BlockQ1_0G128,
    /// Number of block elements.
    len: usize,
    /// Layout used for deallocation.
    layout: Layout,
}

// SAFETY: The buffer owns its allocation exclusively.
unsafe impl Send for AlignedBlocks {}
// SAFETY: Shared references to the buffer are read-only.
unsafe impl Sync for AlignedBlocks {}

impl AlignedBlocks {
    /// Allocate a new zero-initialized aligned buffer of `len` blocks.
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: ALIGNMENT as *mut BlockQ1_0G128,
                len: 0,
                layout: Layout::from_size_align(0, ALIGNMENT)
                    .expect("zero-size layout should always be valid"),
            };
        }

        // Guard against `usize` wraparound (see `AlignedBuffer::new`): an oversized
        // `len` must not silently wrap to an undersized `byte_size`.
        let byte_size = len
            .checked_mul(std::mem::size_of::<BlockQ1_0G128>())
            .expect("AlignedBlocks allocation size overflows usize");
        let layout = Layout::from_size_align(byte_size, ALIGNMENT)
            .expect("layout should be valid for reasonable buffer sizes");

        // SAFETY: layout has non-zero size and valid alignment.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        Self {
            ptr: ptr.cast::<BlockQ1_0G128>(),
            len,
            layout,
        }
    }

    /// Returns the number of blocks.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains no blocks.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the raw pointer to the start of the buffer.
    #[inline]
    pub fn as_ptr(&self) -> *const BlockQ1_0G128 {
        self.ptr
    }

    /// View the buffer as an immutable slice.
    #[inline]
    pub fn as_slice(&self) -> &[BlockQ1_0G128] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for len elements, properly aligned, and zero-initialized.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// View the buffer as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [BlockQ1_0G128] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: ptr is valid for len elements, properly aligned, and we have exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBlocks {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: ptr was allocated with this layout in `new`.
            unsafe {
                std::alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
            }
        }
    }
}

impl std::fmt::Debug for AlignedBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBlocks")
            .field("len", &self.len)
            .field("alignment", &ALIGNMENT)
            .finish()
    }
}

/// Split a slice at cache-line boundaries.
///
/// Returns `(prefix, aligned_middle, suffix)` where `aligned_middle` starts
/// at a 64-byte aligned address and has a 64-byte aligned length (in bytes).
///
/// If the input is already aligned, `prefix` will be empty.
/// If the input is too short to contain any aligned portion, the entire
/// slice is returned as `prefix` with empty `aligned_middle` and `suffix`.
pub fn align_to_cache_line(data: &[f32]) -> (&[f32], &[f32], &[f32]) {
    if data.is_empty() {
        return (&[], &[], &[]);
    }

    let ptr = data.as_ptr() as usize;
    let f32_size = std::mem::size_of::<f32>();

    // How many bytes past the last alignment boundary?
    let misalign_bytes = ptr % ALIGNMENT;

    // Number of f32s to skip to reach alignment
    let prefix_len = if misalign_bytes == 0 {
        0
    } else {
        let skip_bytes = ALIGNMENT - misalign_bytes;
        // Round up to whole f32s
        skip_bytes.div_ceil(f32_size)
    };

    if prefix_len >= data.len() {
        // Entire slice is in the prefix — no aligned middle
        return (data, &[], &[]);
    }

    let remaining = data.len() - prefix_len;

    // How many f32s fit in a cache-line-aligned chunk?
    let f32s_per_line = ALIGNMENT / f32_size; // 16
    let aligned_len = (remaining / f32s_per_line) * f32s_per_line;

    let prefix = &data[..prefix_len];
    let aligned = &data[prefix_len..prefix_len + aligned_len];
    let suffix = &data[prefix_len + aligned_len..];

    (prefix, aligned, suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buffer_new_and_access() {
        let buf = AlignedBuffer::new(128);
        assert_eq!(buf.len(), 128);
        assert!(!buf.is_empty());
        // All zeros
        for &v in buf.as_slice() {
            assert!((v - 0.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn aligned_buffer_alignment() {
        let buf = AlignedBuffer::new(256);
        let ptr_val = buf.as_ptr() as usize;
        assert_eq!(
            ptr_val % ALIGNMENT,
            0,
            "buffer pointer {ptr_val:#x} is not 64-byte aligned"
        );
    }

    #[test]
    fn aligned_buffer_zero_length() {
        let buf = AlignedBuffer::new(0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.as_slice().len(), 0);
    }

    #[test]
    fn aligned_buffer_large() {
        let buf = AlignedBuffer::new(10_000);
        assert_eq!(buf.len(), 10_000);
        assert_eq!(buf.as_ptr() as usize % ALIGNMENT, 0);
    }

    #[test]
    fn aligned_buffer_copy_from_slice() {
        let mut buf = AlignedBuffer::new(8);
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        buf.copy_from_slice(&src);
        assert_eq!(buf.as_slice(), &src);
    }

    #[test]
    fn aligned_buffer_mut_slice() {
        let mut buf = AlignedBuffer::new(4);
        {
            let s = buf.as_mut_slice();
            s[0] = 42.0;
            s[3] = -1.0;
        }
        assert!((buf.as_slice()[0] - 42.0).abs() < f32::EPSILON);
        assert!((buf.as_slice()[3] - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn aligned_blocks_new_and_access() {
        let blocks = AlignedBlocks::new(16);
        assert_eq!(blocks.len(), 16);
        assert!(!blocks.is_empty());
        assert_eq!(blocks.as_ptr() as usize % ALIGNMENT, 0);
    }

    #[test]
    fn aligned_blocks_zero_length() {
        let blocks = AlignedBlocks::new(0);
        assert_eq!(blocks.len(), 0);
        assert!(blocks.is_empty());
        assert_eq!(blocks.as_slice().len(), 0);
    }

    #[test]
    fn align_to_cache_line_empty() {
        let data: &[f32] = &[];
        let (prefix, aligned, suffix) = align_to_cache_line(data);
        assert!(prefix.is_empty());
        assert!(aligned.is_empty());
        assert!(suffix.is_empty());
    }

    #[test]
    fn align_to_cache_line_already_aligned() {
        let buf = AlignedBuffer::new(64);
        let data = buf.as_slice();
        let (prefix, aligned, suffix) = align_to_cache_line(data);
        // Already aligned, so prefix should be empty
        assert!(
            prefix.is_empty(),
            "prefix should be empty for aligned buffer"
        );
        assert_eq!(aligned.len() + suffix.len(), data.len());
    }

    #[test]
    fn align_to_cache_line_preserves_data() {
        let buf = AlignedBuffer::new(128);
        let data = buf.as_slice();
        let (prefix, aligned, suffix) = align_to_cache_line(data);
        // Total length preserved
        assert_eq!(
            prefix.len() + aligned.len() + suffix.len(),
            data.len(),
            "split must preserve total length"
        );
    }

    #[test]
    fn aligned_buffer_debug() {
        let buf = AlignedBuffer::new(32);
        let dbg = format!("{buf:?}");
        assert!(dbg.contains("AlignedBuffer"));
        assert!(dbg.contains("32"));
    }

    #[test]
    fn aligned_blocks_debug() {
        let blocks = AlignedBlocks::new(8);
        let dbg = format!("{blocks:?}");
        assert!(dbg.contains("AlignedBlocks"));
    }

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn aligned_buffer_size_overflow_panics() {
        // `len * size_of::<f32>()` (= 4) overflows `usize`. The constructor must
        // panic loudly instead of silently wrapping to a tiny allocation while
        // `self.len` records the huge original length (which would make
        // `as_slice()` hand out an out-of-bounds slice — undefined behavior).
        let _ = AlignedBuffer::new(usize::MAX / 2);
    }

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn aligned_blocks_size_overflow_panics() {
        // Same wraparound guard for the block buffer (`size_of::<BlockQ1_0G128>()`).
        let _ = AlignedBlocks::new(usize::MAX / 2);
    }
}
