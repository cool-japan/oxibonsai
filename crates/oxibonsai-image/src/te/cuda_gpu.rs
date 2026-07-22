//! GPU (CUDA) backend for the FLUX.2 text-encoder (Qwen3-4B) f32 matmuls.
//!
//! CUDA sibling of `crate::te::gpu` (the Metal backend), authored as a
//! line-for-line mirror. It routes the dominant per-layer Linears of the
//! Qwen3-4B text encoder (Q/K/V/o_proj + gate/up/down across 36 layers) onto the
//! project's f32-exact CUDA GEMM kernel (`CudaGraph::encode_gemm_f32` in
//! `oxibonsai-kernels`), keeping each weight's row-major f32 bytes resident on
//! the GPU and crossing the bus only with the (small) f32 activations per
//! matmul.
//!
//! Unlike the DiT path ([`crate::cuda_gpu`]), the TE weights are **pure f32**
//! (the 4-bit MLX weights are dequantized to f32 offline by `TeWeights`), so the
//! op is a plain `out[m,n] = Σ_k input[m,k] · weight[n,k]` with **no
//! quantization** — the GPU only reassociates the sum, which keeps it cos ≈ 1.0
//! vs the CPU `gemm_abt` reference. Parity is therefore trivially safe (the
//! `te_parity` gate stays cos ≥ 0.999).
//!
//! The whole module is gated on `cfg(all(feature = "native-cuda", any(target_os
//! = "linux", target_os = "windows")))` — the same gate under which
//! `oxibonsai-kernels` re-exports `CudaGraph` — and is `target_os`-DISJOINT
//! from the Metal gate (macOS), so a non-CUDA build never references it and the
//! default Pure-Rust CPU path is entirely unaffected.
//!
//! Default OFF: unlike the DiT (`OXI_DIT_GPU`, default ON), the TE GPU path is
//! opt-in via `OXI_TE_GPU=1` (the same env var as the Metal path; Metal and CUDA
//! are mutually exclusive at build by `target_os`). The CPU TE already tracks the
//! goldens; the GPU path is a speed optimization, enabled explicitly for A/B and
//! production use.
//!
//! On *any* error this module returns a `CudaTeGpuMatmulError`; the caller (the
//! `matmul` helper in [`crate::te::forward`]) swallows it and falls back to the
//! CPU [`crate::gemm::gemm_abt`], so a GPU failure can never break a forward
//! pass (no `unwrap`/`expect`/`panic!`).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use oxibonsai_kernels::{CudaGraph, CudaGraphError};

/// An error from the GPU f32-matmul path. The caller converts this into a
/// silent CPU fallback, so it never propagates out of a forward pass.
#[derive(Debug, thiserror::Error)]
pub enum CudaTeGpuMatmulError {
    /// The process-wide CUDA graph singleton could not be obtained (e.g. no
    /// CUDA device, or the device failed to initialise).
    #[error("CUDA graph unavailable: {0}")]
    GraphUnavailable(String),
    /// The f32-exact CUDA GEMM (weight upload / encode / dispatch) failed.
    #[error("CUDA f32 GEMM failed: {0}")]
    Cuda(#[from] CudaGraphError),
}

/// One-time confirmation that the TE GPU path actually executed at least once
/// (used by the parity example to PROVE the GPU ran, not a silent CPU fallback).
static TE_GPU_USED: AtomicBool = AtomicBool::new(false);

/// Returns `true` once any [`te_matmul_gpu`] call has succeeded.
///
/// Lock-free and cheap; intended for diagnostics / parity assertions.
pub fn te_gpu_was_used() -> bool {
    TE_GPU_USED.load(Ordering::Relaxed)
}

/// Cached runtime toggle for the GPU TE path. Default **OFF**; set env
/// `OXI_TE_GPU=1` to route the TE matmuls through the CUDA f32 GEMM.
static TE_GPU_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether the text encoder should use the GPU f32 path.
///
/// `false` unless the environment variable `OXI_TE_GPU` is set to `1`. The env
/// read is cached in a [`OnceLock`] on first call.
pub fn te_gpu_enabled() -> bool {
    *TE_GPU_ENABLED.get_or_init(|| matches!(std::env::var("OXI_TE_GPU").ok().as_deref(), Some("1")))
}

/// Device-weight residency byte budget, read once from
/// `OXI_TE_GPU_RESIDENT_BUDGET_MB`.
///
/// **Default 0 (disabled).** On a discrete GPU the ~16 GB f32 encoder cannot be
/// held resident alongside the DiT/VAE, so residency is opt-in and VRAM-bounded
/// rather than unconditional (unlike the Metal path on Apple unified memory).
/// A value of 0 preserves the original evict-after-every-GEMM behaviour (the
/// one-shot CLI's low-VRAM profile); a positive `N` MB lets a resident
/// [`crate::session::ImageSession`] keep up to `N` MB of TE device weights cached
/// across prompts (LRU-evicted beyond the budget), so large-VRAM machines
/// amortize the upload across renders while smaller ones stay safe.
static TE_RESIDENT_BUDGET_BYTES: OnceLock<u64> = OnceLock::new();

/// The configured device-weight residency budget in bytes (0 = disabled).
fn resident_budget_bytes() -> u64 {
    *TE_RESIDENT_BUDGET_BYTES.get_or_init(|| {
        std::env::var("OXI_TE_GPU_RESIDENT_BUDGET_MB")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(0)
    })
}

/// Bounded LRU tracker for GPU-resident TE weights, keyed by the (stable, while
/// resident) host weight pointer. It records which device-cache keys are held
/// and their byte sizes, and — on each use — reports which keys must be evicted
/// to stay within the configured budget.
///
/// This layer sits on top of the kernel's unbounded `f32_weight_cache`: the image
/// crate owns the eviction policy so it can cap on-device memory (the kernel API
/// only offers upload / evict primitives).
#[derive(Default)]
struct ResidencyLru {
    /// Sum of the byte sizes of every currently-resident key.
    total: u64,
    /// key → device-buffer byte size.
    sizes: HashMap<u64, u64>,
    /// LRU order: front = least-recently-used, back = most-recently-used.
    order: VecDeque<u64>,
}

impl ResidencyLru {
    /// Mark `key` (`bytes`) as most-recently-used and return the keys that should
    /// be evicted so the resident total stays within `budget`.
    ///
    /// The just-used `key` is never evicted while any older key remains (LRU
    /// evicts from the front); if `key` is the *only* resident entry and alone
    /// exceeds `budget`, it is dropped too — the net effect is then identical to
    /// the non-resident evict-after-GEMM path (nothing persists). Pure and
    /// deterministic (no GPU), so it is unit-testable without a device.
    fn touch(&mut self, key: u64, bytes: u64, budget: u64) -> Vec<u64> {
        // `order` and `sizes` stay in sync (a key is in one iff in the other), so
        // the position lookup doubles as the membership test.
        match self.order.iter().position(|&k| k == key) {
            // Already resident: move it to the most-recent slot (size unchanged).
            Some(pos) => {
                self.order.remove(pos);
            }
            // New key: record its device-buffer size and count it toward total.
            None => {
                self.sizes.insert(key, bytes);
                self.total = self.total.saturating_add(bytes);
            }
        }
        self.order.push_back(key);

        let mut evicted = Vec::new();
        while self.total > budget && self.order.len() > 1 {
            let Some(front) = self.order.pop_front() else {
                break;
            };
            if let Some(sz) = self.sizes.remove(&front) {
                self.total = self.total.saturating_sub(sz);
            }
            evicted.push(front);
        }
        // Only the just-used key remains but it alone exceeds the budget: drop it
        // (do not persist a single weight larger than the whole budget).
        if self.total > budget && self.order.len() == 1 {
            if let Some(front) = self.order.pop_front() {
                if let Some(sz) = self.sizes.remove(&front) {
                    self.total = self.total.saturating_sub(sz);
                }
                evicted.push(front);
            }
        }
        evicted
    }
}

/// The process-wide residency tracker (only consulted on the resident path).
static TE_RESIDENCY: OnceLock<Mutex<ResidencyLru>> = OnceLock::new();

fn residency() -> &'static Mutex<ResidencyLru> {
    TE_RESIDENCY.get_or_init(|| Mutex::new(ResidencyLru::default()))
}

/// Compute `out[m, n] = Σ_k input[m, k] · weight[n, k]` (`x · Wᵀ`) on the GPU.
///
/// - `weight`: row-major f32 `[n, k]` (the dequantized TE Linear weight,
///   borrowed from the long-lived [`crate::te::weights::TeWeights`] registry).
/// - `input`: row-major `[m, k]`.
/// - `out`: row-major `[m, n]` (written in full).
/// - `resident`: whether the host weight buffer is held resident (stable
///   `as_ptr()` identity — [`crate::te::weights::TeWeights::is_resident`]).
///
/// ## Residency policy (CUDA, discrete VRAM)
///
/// When `resident` is `false` (the one-shot CLI's RAM-frugal `Mlx4bit` no-cache
/// policy), the dequantised buffer is freed after this call and the allocator can
/// recycle its address, so the pointer key is unstable — the device buffer is
/// **evicted immediately** after the GEMM (every matmul re-uploads fresh; only
/// one TE weight is GPU-resident at a time). This is byte-for-byte the original
/// behaviour and preserves the low-VRAM profile.
///
/// When `resident` is `true` **and** a positive residency budget is configured
/// (`OXI_TE_GPU_RESIDENT_BUDGET_MB`), the uploaded device buffer is kept across
/// calls under a bounded LRU (the pointer is stable because the resident
/// [`crate::te::weights::TeWeights`] cache holds the `Rc<Tensor>` alive for the
/// whole registry lifetime, giving each weight a unique, lasting key). Weights
/// are LRU-evicted once the resident total exceeds the budget, so on-device
/// memory never grows without bound — the amortization only materialises on a
/// GPU with enough free VRAM to hold a meaningful slice of the encoder. With no
/// budget configured (the default) the resident path also evicts every call, so
/// enabling `OXI_TE_GPU` on a 16 GB card never risks an OOM by default.
///
/// # Errors
/// Returns `CudaTeGpuMatmulError` if the CUDA graph is unavailable or the
/// kernel upload/encode fails (e.g. a length mismatch). The caller falls back to
/// the CPU path on any error.
pub fn te_matmul_gpu(
    weight: &[f32],
    input: &[f32],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    resident: bool,
) -> Result<(), CudaTeGpuMatmulError> {
    let graph =
        CudaGraph::global().map_err(|e| CudaTeGpuMatmulError::GraphUnavailable(e.to_string()))?;
    // The pointer is a valid cache key only when it is a *stable* identity. When
    // `resident`, the `TeWeights` cache owns the `Rc<Tensor>` for the whole
    // registry lifetime, so `as_ptr()` is unique and lasting per weight. When not
    // resident, the dequant buffer is freed after this call and the address is
    // recycled, so the key is only sound for THIS call and must be evicted after
    // the GEMM (else a later Linear at the same recycled address collides and
    // `get_or_upload_f32_weight` hands back a *stale* buffer → wrong weights).
    let key = weight.as_ptr() as u64;
    let handle = graph.get_or_upload_f32_weight(key, weight)?;
    graph.encode_gemm_f32(&handle, input, out, m, n, k)?;

    let budget = resident_budget_bytes();
    if resident && budget > 0 {
        // Persist under a bounded LRU. `handle` (an `Arc` clone) already keeps the
        // device buffer alive across `encode_gemm_f32`; here we only decide which
        // *older* resident keys to release so the total stays within budget.
        let bytes = (n as u64).saturating_mul(k as u64).saturating_mul(4);
        let to_evict = match residency().lock() {
            Ok(mut lru) => lru.touch(key, bytes, budget),
            // A poisoned lock must never break correctness: fall back to evicting
            // this key (the safe, memory-frugal default).
            Err(_) => vec![key],
        };
        for ev in to_evict {
            // Eviction failure is non-fatal: the next upload overwrites the entry.
            let _ = graph.evict_f32_weight(ev);
        }
    } else {
        // Non-resident (or residency disabled): evict this key immediately.
        graph.evict_f32_weight(key)?;
    }

    TE_GPU_USED.store(true, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn te_gpu_disabled_by_default_when_env_unset() {
        // Note: OnceLock caches the first read; this asserts the default policy
        // (env `OXI_TE_GPU` unset → disabled). It does not mutate the env.
        if std::env::var("OXI_TE_GPU").is_err() {
            assert!(!te_gpu_enabled());
        }
    }

    /// The just-used key is never evicted while an older key can be released
    /// instead, and total stays within budget once older keys are dropped.
    #[test]
    fn residency_lru_evicts_oldest_first_within_budget() {
        let mut lru = ResidencyLru::default();
        // Budget holds two 100-byte weights.
        let budget = 250u64;
        assert!(lru.touch(1, 100, budget).is_empty(), "first fits");
        assert!(lru.touch(2, 100, budget).is_empty(), "second fits");
        // Third pushes total to 300 > 250 → evict the oldest (key 1).
        assert_eq!(lru.touch(3, 100, budget), vec![1]);
        assert_eq!(lru.total, 200);
        // Re-touching key 2 must NOT evict it (it becomes most-recent); key 3
        // stays. Total already within budget → no evictions.
        assert!(lru.touch(2, 100, budget).is_empty());
        // A new key 4 now evicts the least-recent, which is key 3.
        assert_eq!(lru.touch(4, 100, budget), vec![3]);
    }

    /// A single weight larger than the whole budget is not persisted (dropped),
    /// matching the non-resident evict-after-GEMM behaviour for that weight.
    #[test]
    fn residency_lru_drops_single_oversized_weight() {
        let mut lru = ResidencyLru::default();
        let evicted = lru.touch(7, 500, 100);
        assert_eq!(evicted, vec![7], "oversized single weight is not kept");
        assert_eq!(lru.total, 0);
        assert!(lru.order.is_empty());
    }

    /// A zero budget (the default) evicts every key immediately, so nothing ever
    /// accumulates on-device.
    #[test]
    fn residency_lru_zero_budget_persists_nothing() {
        let mut lru = ResidencyLru::default();
        assert_eq!(lru.touch(1, 64, 0), vec![1]);
        assert_eq!(lru.touch(2, 64, 0), vec![2]);
        assert_eq!(lru.total, 0);
    }

    /// Residency is disabled by default (no budget env → 0 bytes), so the CUDA
    /// path keeps its safe evict-after-every-GEMM behaviour out of the box.
    #[test]
    fn resident_budget_defaults_to_disabled_when_env_unset() {
        if std::env::var("OXI_TE_GPU_RESIDENT_BUDGET_MB").is_err() {
            assert_eq!(resident_budget_bytes(), 0);
        }
    }
}
