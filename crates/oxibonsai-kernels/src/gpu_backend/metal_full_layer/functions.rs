//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub(super) mod gpu_profile {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use std::time::Instant;
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static INITIALIZED: AtomicBool = AtomicBool::new(false);
    pub struct GpuTimings {
        pub wall_ms: f64,
        pub cpu_encode_ms: f64,
        pub gpu_exec_ms: f64,
    }
    static STATS: Mutex<Vec<GpuTimings>> = Mutex::new(Vec::new());
    pub fn is_enabled() -> bool {
        if !INITIALIZED.load(Ordering::Relaxed) {
            INITIALIZED.store(true, Ordering::Relaxed);
            if std::env::var("OXIBONSAI_PROFILE_GPU").is_ok() {
                ENABLED.store(true, Ordering::Relaxed);
            }
        }
        ENABLED.load(Ordering::Relaxed)
    }
    /// Whether full per-layer/tail profiling output (`OXIBONSAI_PROFILE`,
    /// distinct from the GPU-timing-only `OXIBONSAI_PROFILE_GPU` behind
    /// [`is_enabled`]) is enabled, cached on first call.
    ///
    /// MET-15b: `std::env::var("OXIBONSAI_PROFILE").is_ok()` used to run
    /// **per forward call** (i.e. per decoded token) at both call sites in
    /// `functions_2.rs::{encode_full_forward, encode_full_forward_ternary}`.
    /// `std::env::var` takes the process-wide environment lock and
    /// allocates a `String` on every call; hoisting it into a `OnceLock<bool>`
    /// (mirroring [`is_enabled`]'s `AtomicBool` cache) makes steady-state
    /// decode read a single relaxed atomic instead.
    pub fn full_profiling_enabled() -> bool {
        static FULL_PROFILING: OnceLock<bool> = OnceLock::new();
        *FULL_PROFILING.get_or_init(|| std::env::var("OXIBONSAI_PROFILE").is_ok())
    }
    /// Get GPU execution start/end times from a completed command buffer.
    ///
    /// Uses Objective-C `GPUStartTime` / `GPUEndTime` properties (CFTimeInterval,
    /// seconds since boot). Available on macOS 10.15+.
    ///
    /// # Safety
    /// Must be called only after `wait_until_completed()` returns.
    pub unsafe fn gpu_cmd_times(cmd_buf: &metal::CommandBufferRef) -> (f64, f64) {
        // `deps-17`: `metal` 0.33 re-exports `objc` (`pub extern crate objc;`),
        // so these macros come from the `metal` dependency and the crate needs
        // no direct `objc` dependency of its own. `msg_send!` expands to
        // `sel!`/`sel_impl!`, hence all three imports.
        use metal::objc::{msg_send, sel, sel_impl};
        let start: f64 = msg_send![cmd_buf, GPUStartTime];
        let end: f64 = msg_send![cmd_buf, GPUEndTime];
        (start, end)
    }
    pub fn record_and_print(
        wall_start: Instant,
        encode_end: Instant,
        gpu_start: f64,
        gpu_end: f64,
    ) {
        let wall_end = Instant::now();
        let wall_ms = wall_end.duration_since(wall_start).as_secs_f64() * 1000.0;
        let cpu_encode_ms = encode_end.duration_since(wall_start).as_secs_f64() * 1000.0;
        let gpu_exec_ms = (gpu_end - gpu_start) * 1000.0;
        let overhead_ms = (wall_ms - cpu_encode_ms - gpu_exec_ms).max(0.0);
        if let Ok(mut stats) = STATS.lock() {
            let token_num = stats.len();
            eprintln!(
                "[GPU Profile] token={} wall={:.1}ms cpu_encode={:.1}ms gpu_exec={:.1}ms overhead={:.1}ms",
                token_num, wall_ms, cpu_encode_ms, gpu_exec_ms, overhead_ms,
            );
            stats.push(GpuTimings {
                wall_ms,
                cpu_encode_ms,
                gpu_exec_ms,
            });
        }
    }
    pub fn print_summary(model_size_bytes: u64) {
        if let Ok(stats) = STATS.lock() {
            if stats.is_empty() {
                return;
            }
            let n = stats.len() as f64;
            let avg_wall: f64 = stats.iter().map(|s| s.wall_ms).sum::<f64>() / n;
            let avg_cpu: f64 = stats.iter().map(|s| s.cpu_encode_ms).sum::<f64>() / n;
            let avg_gpu: f64 = stats.iter().map(|s| s.gpu_exec_ms).sum::<f64>() / n;
            let avg_overhead = (avg_wall - avg_cpu - avg_gpu).max(0.0);
            let gpu_bw = if avg_gpu > 0.0 {
                (model_size_bytes as f64) / (avg_gpu / 1000.0) / 1e9
            } else {
                0.0
            };
            eprintln!(
                "[GPU Profile Summary] tokens={} avg: wall={:.1}ms cpu={:.1}ms gpu={:.1}ms overhead={:.1}ms gpu_bw={:.1}GB/s",
                stats.len() as u64, avg_wall, avg_cpu, avg_gpu, avg_overhead, gpu_bw,
            );
        }
    }
}
/// Print the GPU profiling summary (call at end of generation).
///
/// `model_size_bytes` is the model file size, used to compute effective bandwidth.
/// This is a no-op if `OXIBONSAI_PROFILE_GPU` was not set.
pub fn print_gpu_profile_summary(model_size_bytes: u64) {
    if gpu_profile::is_enabled() {
        gpu_profile::print_summary(model_size_bytes);
    }
}
