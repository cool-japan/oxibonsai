//! Software prefetch hints for GEMV/GEMM kernel operations.
//!
//! Provides platform-abstracted prefetch intrinsics that compile to
//! the appropriate hardware instruction on x86-64 (`_mm_prefetch`) and
//! AArch64 (`__prefetch`), and are no-ops on platforms without support.
//!
//! These hints allow the CPU to begin loading cache lines before they
//! are needed, hiding memory latency in compute-bound loops.

/// Number of blocks to prefetch ahead in GEMV/GEMM loops.
const DEFAULT_LOOKAHEAD_BLOCKS: usize = 4;

/// Configuration for prefetch behavior in kernel loops.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// How many blocks ahead to prefetch in the inner loop.
    /// Higher values hide more latency but consume more cache.
    pub lookahead_blocks: usize,
    /// Which prefetch strategy to use.
    pub strategy: PrefetchStrategy,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            lookahead_blocks: DEFAULT_LOOKAHEAD_BLOCKS,
            strategy: PrefetchStrategy::Temporal,
        }
    }
}

impl PrefetchConfig {
    /// Create a config optimized for GEMV (single vector, temporal reuse of weights).
    pub fn for_gemv() -> Self {
        Self {
            lookahead_blocks: 4,
            strategy: PrefetchStrategy::Temporal,
        }
    }

    /// Create a config optimized for GEMM (batch, streaming weights).
    ///
    /// In GEMM, each weight block is reused across the M dimension,
    /// so temporal locality is still useful. For very large M, however,
    /// the first-touch of weight blocks benefits from non-temporal prefetch
    /// to avoid polluting L1 with data that won't be reused for many iterations.
    pub fn for_gemm(batch_size: usize) -> Self {
        if batch_size > 32 {
            Self {
                lookahead_blocks: 8,
                strategy: PrefetchStrategy::NonTemporal,
            }
        } else {
            Self {
                lookahead_blocks: 4,
                strategy: PrefetchStrategy::Temporal,
            }
        }
    }

    /// No prefetching (baseline for benchmarking).
    pub fn none() -> Self {
        Self {
            lookahead_blocks: 0,
            strategy: PrefetchStrategy::None,
        }
    }
}

/// Prefetch strategy controlling cache line placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchStrategy {
    /// No software prefetch hints issued.
    None,
    /// Prefetch for temporal locality — data goes to L1 cache.
    /// Best when data will be reused soon (e.g., weight blocks reused across batch).
    Temporal,
    /// Prefetch for non-temporal (streaming) access — data goes to L2/L3.
    /// Best when data is used once then evicted (e.g., large streaming loads).
    NonTemporal,
}

/// Prefetch locality hint, controlling which cache level receives the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchLocality {
    /// Data will be reused imminently — prefetch to L1 (closest cache).
    High,
    /// Data might be reused — prefetch to L2.
    Medium,
    /// Data unlikely to be reused — prefetch to L3 or use non-temporal hint.
    Low,
}

/// Issue a software prefetch hint for a read from the given pointer.
///
/// This is a performance hint only — the CPU may ignore it. On platforms
/// without prefetch support, this is a no-op that compiles to nothing.
///
/// # Safety note
///
/// The pointer does not need to be valid (prefetch of invalid addresses
/// is architecturally a no-op on x86 and ARM), but callers should ensure
/// the address is within a reasonable range to avoid TLB pollution.
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T, locality: PrefetchLocality) {
    // x86-64: _mm_prefetch
    #[cfg(target_arch = "x86_64")]
    {
        prefetch_read_x86(ptr.cast::<i8>(), locality);
    }

    // AArch64: __prefetch
    #[cfg(target_arch = "aarch64")]
    {
        prefetch_read_aarch64(ptr.cast::<i8>(), locality);
    }

    // All other platforms: no-op
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
        let _ = locality;
    }
}

/// Issue a software prefetch hint for a write to the given pointer.
///
/// Tells the CPU to fetch the cache line in exclusive/modified state,
/// which avoids a read-for-ownership transaction on the first write.
#[inline(always)]
pub fn prefetch_write<T>(ptr: *mut T, locality: PrefetchLocality) {
    #[cfg(target_arch = "x86_64")]
    {
        prefetch_write_x86(ptr.cast::<i8>(), locality);
    }

    #[cfg(target_arch = "aarch64")]
    {
        prefetch_write_aarch64(ptr.cast::<i8>(), locality);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
        let _ = locality;
    }
}

/// Prefetch a sequence of `count` cache lines starting from `ptr`.
///
/// Useful for prefetching a contiguous array of blocks before processing.
#[inline]
pub fn prefetch_range_read<T>(ptr: *const T, byte_count: usize, locality: PrefetchLocality) {
    let cache_line = 64usize;
    let mut offset = 0;
    while offset < byte_count {
        // SAFETY: We're only issuing prefetch hints; invalid addresses are safe.
        let addr = unsafe { (ptr as *const u8).add(offset) };
        prefetch_read(addr, locality);
        offset += cache_line;
    }
}

// ── x86-64 implementation ───────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_read_x86(ptr: *const i8, locality: PrefetchLocality) {
    // SAFETY: _mm_prefetch is always safe — invalid addresses are silently ignored.
    unsafe {
        match locality {
            PrefetchLocality::High => {
                core::arch::x86_64::_mm_prefetch(ptr, core::arch::x86_64::_MM_HINT_T0);
            }
            PrefetchLocality::Medium => {
                core::arch::x86_64::_mm_prefetch(ptr, core::arch::x86_64::_MM_HINT_T1);
            }
            PrefetchLocality::Low => {
                core::arch::x86_64::_mm_prefetch(ptr, core::arch::x86_64::_MM_HINT_NTA);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_write_x86(ptr: *const i8, locality: PrefetchLocality) {
    // x86 doesn't have a separate write prefetch in SSE — use PREFETCHW if available,
    // otherwise fall back to read prefetch (which still helps).
    // _mm_prefetch with _MM_HINT_ET0 is PREFETCHW (exclusive for write).
    // Not all x86 CPUs support it, so we use read prefetch as a safe fallback.
    prefetch_read_x86(ptr, locality);
}

// ── AArch64 implementation ──────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn prefetch_read_aarch64(ptr: *const i8, locality: PrefetchLocality) {
    // SAFETY: __prefetch is safe — invalid addresses are silently ignored on ARM.
    // AArch64 _prefetch requires const arguments, so we match and call separately.
    // The `aarch64_prefetch!` macro supplies the `unsafe` block (and degrades to
    // a no-op off-nightly, where the intrinsic is unavailable).
    match locality {
        PrefetchLocality::High => {
            crate::aarch64_prefetch!(ptr, 0, 3); // keep in all caches
        }
        PrefetchLocality::Medium => {
            crate::aarch64_prefetch!(ptr, 0, 2); // keep in L2+
        }
        PrefetchLocality::Low => {
            crate::aarch64_prefetch!(ptr, 0, 0); // non-temporal
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn prefetch_write_aarch64(ptr: *const i8, locality: PrefetchLocality) {
    // SAFETY: rw=1 for write/store prefetch. Const arguments required.
    // The `aarch64_prefetch!` macro supplies the `unsafe` block.
    match locality {
        PrefetchLocality::High => {
            crate::aarch64_prefetch!(ptr, 1, 3);
        }
        PrefetchLocality::Medium => {
            crate::aarch64_prefetch!(ptr, 1, 2);
        }
        PrefetchLocality::Low => {
            crate::aarch64_prefetch!(ptr, 1, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_config_defaults() {
        let config = PrefetchConfig::default();
        assert_eq!(config.lookahead_blocks, 4);
        assert_eq!(config.strategy, PrefetchStrategy::Temporal);
    }

    #[test]
    fn prefetch_config_for_gemv() {
        let config = PrefetchConfig::for_gemv();
        assert_eq!(config.strategy, PrefetchStrategy::Temporal);
        assert!(config.lookahead_blocks > 0);
    }

    #[test]
    fn prefetch_config_for_gemm_small_batch() {
        let config = PrefetchConfig::for_gemm(4);
        assert_eq!(config.strategy, PrefetchStrategy::Temporal);
    }

    #[test]
    fn prefetch_config_for_gemm_large_batch() {
        let config = PrefetchConfig::for_gemm(64);
        assert_eq!(config.strategy, PrefetchStrategy::NonTemporal);
        assert!(config.lookahead_blocks > 4);
    }

    #[test]
    fn prefetch_config_none() {
        let config = PrefetchConfig::none();
        assert_eq!(config.lookahead_blocks, 0);
        assert_eq!(config.strategy, PrefetchStrategy::None);
    }

    #[test]
    fn prefetch_read_smoke_test() {
        // Ensure calling prefetch_read doesn't crash
        let data = [1.0f32, 2.0, 3.0, 4.0];
        prefetch_read(data.as_ptr(), PrefetchLocality::High);
        prefetch_read(data.as_ptr(), PrefetchLocality::Medium);
        prefetch_read(data.as_ptr(), PrefetchLocality::Low);
    }

    #[test]
    fn prefetch_write_smoke_test() {
        let mut data = [0.0f32; 16];
        prefetch_write(data.as_mut_ptr(), PrefetchLocality::High);
        prefetch_write(data.as_mut_ptr(), PrefetchLocality::Medium);
        prefetch_write(data.as_mut_ptr(), PrefetchLocality::Low);
        // Should still be writable after prefetch
        data[0] = 42.0;
        assert!((data[0] - 42.0).abs() < f32::EPSILON);
    }

    #[test]
    fn prefetch_range_read_smoke_test() {
        let data = vec![0.0f32; 1024];
        let byte_count = data.len() * std::mem::size_of::<f32>();
        prefetch_range_read(data.as_ptr(), byte_count, PrefetchLocality::High);
        prefetch_range_read(data.as_ptr(), byte_count, PrefetchLocality::Low);
    }

    #[test]
    fn prefetch_strategy_equality() {
        assert_eq!(PrefetchStrategy::None, PrefetchStrategy::None);
        assert_ne!(PrefetchStrategy::Temporal, PrefetchStrategy::NonTemporal);
    }

    #[test]
    fn prefetch_locality_equality() {
        assert_eq!(PrefetchLocality::High, PrefetchLocality::High);
        assert_ne!(PrefetchLocality::High, PrefetchLocality::Low);
    }
}
