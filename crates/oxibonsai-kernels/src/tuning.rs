//! Platform-aware performance tuning for kernel dispatch.
//!
//! Detects hardware characteristics (core counts, cache sizes, SIMD capabilities)
//! at runtime and computes optimal thresholds for parallel dispatch decisions,
//! tiled GEMM block sizes, and prefetch strategies.

use std::sync::OnceLock;

/// Group size for BlockQ1_0G128 (128 weights per block).
const BLOCK_GROUP_SIZE: usize = 128;

/// Global cached platform profile, detected once on first access.
static GLOBAL_PROFILE: OnceLock<PlatformProfile> = OnceLock::new();

/// Global cached tuned thresholds, computed once from the platform profile.
static GLOBAL_THRESHOLDS: OnceLock<TunedThresholds> = OnceLock::new();

/// Platform characteristics detected at init time.
///
/// Captures hardware topology (core counts, cache hierarchy) and SIMD
/// capability flags for the current CPU. Used to derive optimal
/// parallelism thresholds and tiling parameters.
#[derive(Debug, Clone)]
pub struct PlatformProfile {
    /// Number of logical (hardware thread) cores.
    pub logical_cores: usize,
    /// Number of physical cores (estimated; logical/2 on x86 HT, = logical on ARM).
    pub physical_cores: usize,
    /// Cache line size in bytes (typically 64 on modern CPUs).
    pub cache_line_bytes: usize,
    /// Estimated L1 data cache size per core in bytes.
    pub l1_cache_bytes: usize,
    /// Estimated L2 cache size per core in bytes.
    pub l2_cache_bytes: usize,
    /// Whether AVX2 (256-bit SIMD) is available (x86-64 only).
    pub has_avx2: bool,
    /// Whether AVX-512 (512-bit SIMD) is available (x86-64 only).
    pub has_avx512: bool,
    /// Whether NEON (128-bit SIMD) is available (AArch64 only).
    pub has_neon: bool,
}

/// Tuned thresholds for parallel dispatch and tiling.
///
/// These values are derived from the [`PlatformProfile`] and control
/// when parallel execution is engaged and how tiled GEMM partitions work.
#[derive(Debug, Clone)]
pub struct TunedThresholds {
    /// Minimum number of output rows before parallel GEMV is engaged.
    /// Below this, sequential execution wins due to lower overhead.
    pub par_gemv_min_rows: usize,
    /// Minimum batch size before parallel GEMM is engaged.
    pub par_gemm_min_batch: usize,
    /// Minimum number of output rows before cache-tiled parallel dispatch
    /// (L1/L2-aware) is preferred over flat row-parallel dispatch.
    /// Always >= [`Self::par_gemv_min_rows`], so the Direct -> ParallelRow ->
    /// ParallelTiled tiers used by [`crate::parallel_tiled::select_gemv_strategy`]
    /// stay correctly ordered regardless of platform.
    pub par_tiled_min_rows: usize,
    /// Block size along M (rows) for tiled GEMM.
    pub tiled_gemm_block_m: usize,
    /// Block size along N (columns) for tiled GEMM.
    pub tiled_gemm_block_n: usize,
    /// Block size along K (inner/reduction) for tiled GEMM.
    /// Always a multiple of the block group size (128).
    pub tiled_gemm_block_k: usize,
}

impl PlatformProfile {
    /// Detect current platform capabilities.
    ///
    /// Uses `std::thread::available_parallelism()` for core counts and
    /// compile-time/runtime feature detection for SIMD flags. Cache sizes
    /// are estimated from typical values for the detected architecture.
    pub fn detect() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        let physical_cores = Self::estimate_physical_cores(logical_cores);
        let cache_line_bytes = 64; // Universal on modern x86-64 and AArch64

        // L1/L2 estimates per core — conservative defaults
        let (l1_cache_bytes, l2_cache_bytes) = Self::estimate_cache_sizes();

        let has_avx2 = Self::detect_avx2();
        let has_avx512 = Self::detect_avx512();
        let has_neon = Self::detect_neon();

        Self {
            logical_cores,
            physical_cores,
            cache_line_bytes,
            l1_cache_bytes,
            l2_cache_bytes,
            has_avx2,
            has_avx512,
            has_neon,
        }
    }

    /// Get the global cached platform profile (detected once, reused).
    pub fn global() -> &'static PlatformProfile {
        GLOBAL_PROFILE.get_or_init(Self::detect)
    }

    /// Compute optimal thresholds for this platform.
    ///
    /// The logic scales parallel thresholds inversely with core count
    /// (more cores => lower threshold to engage parallelism) and sizes
    /// tiled GEMM blocks to fit in L1 cache.
    pub fn compute_thresholds(&self) -> TunedThresholds {
        let par_gemv_min_rows = self.compute_gemv_threshold();
        let par_gemm_min_batch = self.compute_gemm_threshold();
        let par_tiled_min_rows = self.compute_tiled_min_rows(par_gemv_min_rows);
        let (block_m, block_n, block_k) = self.compute_tile_sizes();

        TunedThresholds {
            par_gemv_min_rows,
            par_gemm_min_batch,
            par_tiled_min_rows,
            tiled_gemm_block_m: block_m,
            tiled_gemm_block_n: block_n,
            tiled_gemm_block_k: block_k,
        }
    }

    /// Get the global cached thresholds (computed once from the global profile).
    pub fn global_thresholds() -> &'static TunedThresholds {
        GLOBAL_THRESHOLDS.get_or_init(|| Self::global().compute_thresholds())
    }

    /// Construct a profile with explicit values (useful for testing).
    pub fn with_cores(logical: usize, physical: usize) -> Self {
        Self {
            logical_cores: logical.max(1),
            physical_cores: physical.max(1),
            cache_line_bytes: 64,
            l1_cache_bytes: 32 * 1024,
            l2_cache_bytes: 256 * 1024,
            has_avx2: false,
            has_avx512: false,
            has_neon: false,
        }
    }

    /// Construct a profile with custom cache sizes (useful for testing).
    pub fn with_cache(l1_bytes: usize, l2_bytes: usize) -> Self {
        Self {
            logical_cores: 4,
            physical_cores: 4,
            cache_line_bytes: 64,
            l1_cache_bytes: l1_bytes,
            l2_cache_bytes: l2_bytes,
            has_avx2: false,
            has_avx512: false,
            has_neon: false,
        }
    }

    // ── Private helpers ──────────────────────────────────────────────

    fn estimate_physical_cores(logical: usize) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            // x86-64: assume hyperthreading (2 threads per physical core)
            (logical / 2).max(1)
        }
        #[cfg(target_arch = "aarch64")]
        {
            // ARM: typically no SMT, logical == physical
            logical
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Conservative: assume no hyperthreading
            logical
        }
    }

    fn estimate_cache_sizes() -> (usize, usize) {
        #[cfg(target_arch = "aarch64")]
        {
            // Apple Silicon and recent ARM server chips often have larger caches
            // M1/M2 P-cores: 192KB L1d, 12MB shared L2 (but per-core share ~1.5MB)
            // Conservative estimate for portability
            (64 * 1024, 512 * 1024)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // Typical x86-64: 32KB L1d, 256KB L2 per core
            (32 * 1024, 256 * 1024)
        }
    }

    fn detect_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            // Runtime check using cpuid — works even when not compiled with +avx2
            if is_x86_feature_detected!("avx2") {
                return true;
            }
            // Fall back to compile-time check
            cfg!(target_feature = "avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn detect_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return true;
            }
            cfg!(target_feature = "avx512f")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn detect_neon() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on AArch64
            true
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }

    /// Compute minimum rows for parallel GEMV.
    ///
    /// Scaling logic:
    /// - 1-2 physical cores: 256 (sequential almost always wins)
    /// - 3-4 cores: 128
    /// - 5-8 cores: 64
    /// - 9-15 cores: 48
    /// - 16+ cores: 32
    ///
    /// SIMD availability shifts the break-even point: faster per-row
    /// computation means you need more rows to amortize thread overhead.
    fn compute_gemv_threshold(&self) -> usize {
        let base = match self.physical_cores {
            0..=2 => 256,
            3..=4 => 128,
            5..=8 => 64,
            9..=15 => 48,
            _ => 32,
        };

        // SIMD makes each row faster, so raise the threshold slightly
        // (parallel overhead becomes relatively more expensive)
        let simd_factor = if self.has_avx512 {
            // AVX-512: very fast per-row, need more rows for parallel to win
            3
        } else if self.has_avx2 || self.has_neon {
            2
        } else {
            1
        };

        // Scale: base * (1 + simd_factor * 0.25), rounded to multiple of 8
        let adjusted = base + (base * simd_factor) / 4;
        round_up_to(adjusted, 8)
    }

    /// Compute minimum batch for parallel GEMM.
    ///
    /// Similar scaling to GEMV but for the batch dimension.
    fn compute_gemm_threshold(&self) -> usize {
        match self.physical_cores {
            0..=2 => 16,
            3..=4 => 8,
            5..=8 => 4,
            9..=15 => 3,
            _ => 2,
        }
    }

    /// Compute the minimum row count to prefer cache-tiled parallel dispatch
    /// (parallel L2 tiles, each internally L1-tiled) over flat row-parallel
    /// dispatch.
    ///
    /// Historically this was a flat constant of 256 rows, sized for a
    /// ~256 KB L2 slice per core. We scale that baseline by the platform's
    /// actual detected L2 cache size, so a chip with a larger L2 (e.g.
    /// Apple Silicon) delays the switch into tiling until there are enough
    /// rows to justify it, while a smaller-cache chip switches sooner.
    ///
    /// The result is clamped to never fall below `par_gemv_min_rows`, which
    /// keeps the Direct -> ParallelRow -> ParallelTiled tiers correctly
    /// ordered even for unusual platform profiles (e.g. very small cache,
    /// many cores).
    fn compute_tiled_min_rows(&self, par_gemv_min_rows: usize) -> usize {
        /// Legacy fixed threshold this scaling is anchored to.
        const BASELINE_TILED_MIN_ROWS: usize = 256;
        /// L2 size the legacy threshold assumed (typical x86-64 per-core L2).
        const BASELINE_L2_BYTES: usize = 256 * 1024;

        let scaled = (BASELINE_TILED_MIN_ROWS as u128 * self.l2_cache_bytes as u128
            / BASELINE_L2_BYTES as u128) as usize;
        let scaled = round_up_to(scaled.max(1), 8);
        scaled.max(par_gemv_min_rows)
    }

    /// Compute tiled GEMM block sizes to fit in L1 cache.
    ///
    /// Strategy: for a tile of size block_m x block_n, we need:
    ///   - block_m * block_k floats of input A
    ///   - block_k * block_n packed weights (much smaller due to 1-bit)
    ///   - block_m * block_n floats of output C
    ///
    /// We want the working set to fit in L1. Since weights are 1-bit packed,
    /// the dominant memory is the f32 input/output tiles.
    ///
    /// Approximate: 3 * block_size^2 * sizeof(f32) <= L1 size
    ///   => block_size = sqrt(L1 / (3 * 4))
    fn compute_tile_sizes(&self) -> (usize, usize, usize) {
        let l1 = self.l1_cache_bytes;
        let sizeof_f32 = std::mem::size_of::<f32>();

        // Leave some L1 headroom (use 75% of L1 for tiles)
        let usable_l1 = (l1 * 3) / 4;

        // block_size = sqrt(usable_l1 / (3 * sizeof(f32)))
        let raw_block = isqrt(usable_l1 / (3 * sizeof_f32));

        // Round down to nearest multiple of 8 for SIMD alignment, minimum 8
        let block_mn = round_down_to(raw_block, 8).max(8);

        // block_k: must be a multiple of the group size (128)
        // Choose smallest multiple of 128 that doesn't exceed the computed block size
        // but at minimum one group.
        let block_k = if block_mn >= BLOCK_GROUP_SIZE {
            round_down_to(block_mn, BLOCK_GROUP_SIZE)
        } else {
            BLOCK_GROUP_SIZE
        };

        (block_mn, block_mn, block_k)
    }
}

impl TunedThresholds {
    /// Check whether a GEMV with the given row count should use parallelism.
    #[inline]
    pub fn should_parallelize_gemv(&self, n_rows: usize) -> bool {
        n_rows >= self.par_gemv_min_rows
    }

    /// Check whether a GEMM with the given batch size should use parallelism.
    #[inline]
    pub fn should_parallelize_gemm(&self, batch_size: usize) -> bool {
        batch_size >= self.par_gemm_min_batch
    }
}

/// Summary of the current platform's tuning configuration.
///
/// Useful for diagnostics and logging during model initialization.
#[derive(Debug, Clone)]
pub struct TuningSummary {
    pub profile: PlatformProfile,
    pub thresholds: TunedThresholds,
}

impl TuningSummary {
    /// Build a summary from the global profile and thresholds.
    pub fn current() -> Self {
        Self {
            profile: PlatformProfile::global().clone(),
            thresholds: PlatformProfile::global_thresholds().clone(),
        }
    }
}

impl std::fmt::Display for TuningSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Platform Tuning Summary")?;
        writeln!(
            f,
            "  Cores: {} logical, {} physical",
            self.profile.logical_cores, self.profile.physical_cores
        )?;
        writeln!(
            f,
            "  Cache: L1={} KB, L2={} KB, line={} B",
            self.profile.l1_cache_bytes / 1024,
            self.profile.l2_cache_bytes / 1024,
            self.profile.cache_line_bytes
        )?;
        writeln!(
            f,
            "  SIMD: AVX2={}, AVX-512={}, NEON={}",
            self.profile.has_avx2, self.profile.has_avx512, self.profile.has_neon
        )?;
        writeln!(f, "  Thresholds:")?;
        writeln!(
            f,
            "    par_gemv_min_rows: {}",
            self.thresholds.par_gemv_min_rows
        )?;
        writeln!(
            f,
            "    par_gemm_min_batch: {}",
            self.thresholds.par_gemm_min_batch
        )?;
        writeln!(
            f,
            "    par_tiled_min_rows: {}",
            self.thresholds.par_tiled_min_rows
        )?;
        writeln!(
            f,
            "    tiled_gemm_block: {}x{}x{}",
            self.thresholds.tiled_gemm_block_m,
            self.thresholds.tiled_gemm_block_n,
            self.thresholds.tiled_gemm_block_k
        )?;
        Ok(())
    }
}

// ── Arithmetic helpers ──────────────────────────────────────────────────

/// Integer square root (floor).
fn isqrt(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Round `v` up to the nearest multiple of `align` (align must be > 0).
fn round_up_to(v: usize, align: usize) -> usize {
    debug_assert!(align > 0);
    let rem = v % align;
    if rem == 0 {
        v
    } else {
        v + (align - rem)
    }
}

/// Round `v` down to the nearest multiple of `align` (align must be > 0).
fn round_down_to(v: usize, align: usize) -> usize {
    debug_assert!(align > 0);
    v - (v % align)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_valid_profile() {
        let profile = PlatformProfile::detect();
        assert!(
            profile.logical_cores > 0,
            "must have at least 1 logical core"
        );
        assert!(
            profile.physical_cores > 0,
            "must have at least 1 physical core"
        );
        assert!(profile.physical_cores <= profile.logical_cores);
        assert_eq!(profile.cache_line_bytes, 64);
        assert!(
            profile.l1_cache_bytes >= 16 * 1024,
            "L1 should be at least 16KB"
        );
        assert!(
            profile.l2_cache_bytes >= 64 * 1024,
            "L2 should be at least 64KB"
        );
    }

    #[test]
    fn thresholds_are_reasonable() {
        let profile = PlatformProfile::detect();
        let thresholds = profile.compute_thresholds();
        assert!(thresholds.par_gemv_min_rows >= 8);
        assert!(thresholds.par_gemv_min_rows <= 1024);
        assert!(thresholds.par_gemm_min_batch >= 1);
        assert!(thresholds.par_gemm_min_batch <= 64);
    }

    #[test]
    fn tile_sizes_multiple_of_8() {
        let profile = PlatformProfile::detect();
        let t = profile.compute_thresholds();
        assert_eq!(t.tiled_gemm_block_m % 8, 0, "block_m must be multiple of 8");
        assert_eq!(t.tiled_gemm_block_n % 8, 0, "block_n must be multiple of 8");
        assert_eq!(
            t.tiled_gemm_block_k % BLOCK_GROUP_SIZE,
            0,
            "block_k must be multiple of group size (128)"
        );
    }

    #[test]
    fn more_cores_lower_gemv_threshold() {
        let p2 = PlatformProfile::with_cores(2, 2);
        let p16 = PlatformProfile::with_cores(16, 16);
        let t2 = p2.compute_thresholds();
        let t16 = p16.compute_thresholds();
        assert!(
            t16.par_gemv_min_rows < t2.par_gemv_min_rows,
            "16 cores ({}) should have lower threshold than 2 cores ({})",
            t16.par_gemv_min_rows,
            t2.par_gemv_min_rows
        );
    }

    #[test]
    fn more_cores_lower_gemm_threshold() {
        let p2 = PlatformProfile::with_cores(2, 2);
        let p16 = PlatformProfile::with_cores(16, 16);
        let t2 = p2.compute_thresholds();
        let t16 = p16.compute_thresholds();
        assert!(
            t16.par_gemm_min_batch < t2.par_gemm_min_batch,
            "16 cores ({}) should have lower threshold than 2 cores ({})",
            t16.par_gemm_min_batch,
            t2.par_gemm_min_batch
        );
    }

    #[test]
    fn isqrt_correctness() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(10), 3); // floor
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(8192), 90); // sqrt(8192) ≈ 90.5
    }

    #[test]
    fn round_helpers() {
        assert_eq!(round_up_to(0, 8), 0);
        assert_eq!(round_up_to(1, 8), 8);
        assert_eq!(round_up_to(8, 8), 8);
        assert_eq!(round_up_to(9, 8), 16);
        assert_eq!(round_down_to(0, 8), 0);
        assert_eq!(round_down_to(7, 8), 0);
        assert_eq!(round_down_to(8, 8), 8);
        assert_eq!(round_down_to(15, 8), 8);
    }

    #[test]
    fn global_profile_consistent() {
        let p1 = PlatformProfile::global();
        let p2 = PlatformProfile::global();
        // Should be the same reference (OnceLock)
        assert_eq!(p1.logical_cores, p2.logical_cores);
        assert_eq!(p1.physical_cores, p2.physical_cores);
    }

    #[test]
    fn tuning_summary_display() {
        let summary = TuningSummary::current();
        let text = format!("{summary}");
        assert!(text.contains("Platform Tuning Summary"));
        assert!(text.contains("Cores:"));
        assert!(text.contains("Cache:"));
        assert!(text.contains("SIMD:"));
    }

    #[test]
    fn should_parallelize_decisions() {
        let p = PlatformProfile::with_cores(8, 8);
        let t = p.compute_thresholds();
        // Below threshold => no parallelism
        assert!(!t.should_parallelize_gemv(1));
        // Above threshold => parallelize
        assert!(t.should_parallelize_gemv(t.par_gemv_min_rows));
        assert!(t.should_parallelize_gemv(t.par_gemv_min_rows + 1));
    }

    #[test]
    fn tiled_min_rows_never_below_gemv_min_rows() {
        // Across a spread of core counts and cache sizes, the tiled-dispatch
        // boundary must never sit below the plain-parallel boundary, or the
        // Direct -> ParallelRow -> ParallelTiled tiers would invert.
        for cores in [1usize, 2, 3, 4, 5, 8, 9, 16, 32] {
            for (l1, l2) in [
                (16 * 1024, 64 * 1024),
                (32 * 1024, 256 * 1024),
                (64 * 1024, 512 * 1024),
                (128 * 1024, 2 * 1024 * 1024),
            ] {
                let mut p = PlatformProfile::with_cores(cores, cores);
                p.l1_cache_bytes = l1;
                p.l2_cache_bytes = l2;
                let t = p.compute_thresholds();
                assert!(
                    t.par_tiled_min_rows >= t.par_gemv_min_rows,
                    "cores={cores} l2={l2}: par_tiled_min_rows ({}) < par_gemv_min_rows ({})",
                    t.par_tiled_min_rows,
                    t.par_gemv_min_rows
                );
            }
        }
    }

    #[test]
    fn tiled_min_rows_scales_with_l2_cache() {
        // Baseline: 256 KB L2 (the historical hardcoded-constant's assumption)
        // should reproduce the historical default of 256 when it doesn't get
        // clamped up by a high par_gemv_min_rows.
        let p_baseline = PlatformProfile::with_cores(4, 4); // with_cores uses 32KB/256KB
        let t_baseline = p_baseline.compute_thresholds();
        assert_eq!(
            t_baseline.par_tiled_min_rows,
            256.max(t_baseline.par_gemv_min_rows)
        );

        // Doubling L2 should not decrease the threshold.
        let p_bigger = PlatformProfile::with_cache(32 * 1024, 512 * 1024);
        let t_bigger = p_bigger.compute_thresholds();
        assert!(t_bigger.par_tiled_min_rows >= t_baseline.par_tiled_min_rows);
    }

    #[test]
    fn with_cache_custom_sizes() {
        let p = PlatformProfile::with_cache(64 * 1024, 1024 * 1024);
        assert_eq!(p.l1_cache_bytes, 64 * 1024);
        assert_eq!(p.l2_cache_bytes, 1024 * 1024);
        let t = p.compute_thresholds();
        // Larger L1 => larger tile sizes
        let p_small = PlatformProfile::with_cache(16 * 1024, 128 * 1024);
        let t_small = p_small.compute_thresholds();
        assert!(
            t.tiled_gemm_block_m >= t_small.tiled_gemm_block_m,
            "larger L1 ({}) should give >= block_m than smaller L1 ({})",
            t.tiled_gemm_block_m,
            t_small.tiled_gemm_block_m
        );
    }
}
