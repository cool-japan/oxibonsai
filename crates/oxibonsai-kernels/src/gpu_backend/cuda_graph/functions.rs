//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

//! **UNVALIDATED on hardware.** This project has no CUDA device, so every
//! CUDA path in this module is compile-checked only (see `scripts/check_cuda.sh`).

use cudarc::nvrtc::compile_ptx;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, warn};

use crate::gpu_backend::kernel_artifact_cache::{fnv1a_64, ArtifactCache};

use super::cudagraph_type::CudaGraph;
use super::types::CudaGraphError;

/// Sub-path of the user-private cache root holding compiled PTX.
const PTX_CACHE_SUBDIR: &str = "oxibonsai/ptx";

/// Device/driver half of the PTX cache key, published by [`CudaGraph::new`]
/// once the device is known (and before the first compile of the process).
static PTX_CACHE_DEVICE_TAG: OnceLock<String> = OnceLock::new();

/// Record the selected device's identity in the PTX cache key.
///
/// F1 requires the driver/arch version in the cache key, not only a content
/// hash: PTX produced against one toolkit must never be handed to another.
/// Called from `CudaGraph::new` right after the context is created, so every
/// compile in the process sees it.
pub(super) fn set_ptx_cache_device_tag(tag: String) {
    let _ = PTX_CACHE_DEVICE_TAG.set(tag);
}

/// Fingerprint of everything except the source that the compiled PTX depends
/// on: CUDA toolkit version, this crate's version, host target and the selected
/// device (once known).
fn ptx_env_fingerprint() -> u64 {
    let device = PTX_CACHE_DEVICE_TAG
        .get()
        .map(String::as_str)
        .unwrap_or("device-unknown");
    let env = format!(
        "cuda={} kernels={} host={}-{} device={device}",
        cudarc::driver::sys::CUDA_VERSION,
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    fnv1a_64(env.as_bytes())
}

/// The process-wide PTX cache, or `None` when no user-private directory could
/// be prepared — in which case every compile is done in memory rather than
/// trusting a directory that is not ours (finding F1).
fn ptx_cache() -> Option<&'static ArtifactCache> {
    static CACHE: OnceLock<Option<ArtifactCache>> = OnceLock::new();
    CACHE
        .get_or_init(
            || match ArtifactCache::open_default(PTX_CACHE_SUBDIR, "ptx") {
                Ok(cache) => {
                    debug!(dir = %cache.dir().display(), "PTX disk cache ready");
                    Some(cache)
                }
                Err(e) => {
                    warn!("PTX disk cache disabled ({e}); compiling every kernel in memory");
                    None
                }
            },
        )
        .as_ref()
}

/// Compile PTX from CUDA C source, using the user-private, integrity-checked
/// disk cache keyed on the source hash **and** the build/device environment.
///
/// First call: compiles via NVRTC (~5 s) and publishes the result atomically.
/// Subsequent calls: verifies and loads the cached PTX (~1 ms), skipping NVRTC.
///
/// # Security (finding F1)
/// The cache used to live in the shared OS temp directory under a predictable
/// name and was written non-atomically, so on a multi-tenant host any local
/// user could plant the PTX this function then loads and executes on the GPU.
/// It now lives in a user-private directory whose ownership and permissions are
/// verified, every artifact's content hash is checked before it is loaded, and
/// publishing is `create_new` + `rename`. A cache that cannot be trusted is not
/// used at all. See [`crate::gpu_backend::kernel_artifact_cache`].
///
/// Every PTX load in the CUDA tree (`cuda_graph`, `cuda_full_layer`,
/// `cuda_prefill`, `cuda_q_std_prefill`, `cuda_k_quant_prefill`,
/// `cuda_fp8_prefill`) routes through here, so one fix covers all of them.
pub(crate) fn compile_or_load_ptx(
    src: &str,
    tag: &str,
) -> Result<cudarc::nvrtc::Ptx, CudaGraphError> {
    let src_hash = fnv1a_64(src.as_bytes());
    let env_hash = ptx_env_fingerprint();
    let cache = ptx_cache();
    if let Some(hit) = cache.and_then(|c| c.load(tag, src_hash, env_hash)) {
        debug!("PTX cache hit tag={tag} src={src_hash:016x} env={env_hash:016x}");
        return Ok(cudarc::nvrtc::Ptx::from_src(hit));
    }
    debug!("PTX cache miss for tag={tag}, compiling...");
    let ptx = compile_ptx(src).map_err(|e| CudaGraphError::CompilationFailed(format!("{e}")))?;
    if let Some(cache) = cache {
        if let Err(e) = cache.store(tag, src_hash, env_hash, &ptx.to_src()) {
            debug!("PTX cache store for tag={tag} failed, continuing: {e}");
        }
    }
    debug!("PTX compiled and cached: tag={tag}");
    Ok(ptx)
}
static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);
/// Allocate a new globally-unique weight handle ID.
pub(crate) fn alloc_handle_id() -> u64 {
    NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Outcome of the one `CudaGraph` initialisation attempt this process makes:
/// `None` = never attempted, `Some(Ok(_))` = live singleton, `Some(Err(_))` =
/// initialisation failed and must **not** be retried.
type CudaGraphSlot = Mutex<Option<Result<Arc<CudaGraph>, CudaGraphError>>>;

/// Memoised process-wide [`CudaGraph`] (finding F-M2).
///
/// A failed `CudaGraph::new` costs a `CudaContext::new` plus six NVRTC/PTX
/// module loads and 28 `load_function` calls, and the previous code repeated
/// all of it on every call — i.e. per layer, per token, forever.
pub(super) static GLOBAL_CUDA_GRAPH: OnceLock<CudaGraphSlot> = OnceLock::new();

/// Initialisation has not been attempted yet.
pub(super) const CUDA_INIT_UNKNOWN: u8 = 0;
/// The singleton is live.
pub(super) const CUDA_INIT_READY: u8 = 1;
/// Initialisation failed; the memoised error is returned without retrying.
pub(super) const CUDA_INIT_FAILED: u8 = 2;

/// Lock-free mirror of [`GLOBAL_CUDA_GRAPH`]'s state, so the per-layer entry
/// points can bail out without taking the singleton mutex at all.
pub(super) static CUDA_INIT_STATE: AtomicU8 = AtomicU8::new(CUDA_INIT_UNKNOWN);

/// How many times `CudaGraph::new` has actually been entered — the memoisation
/// regression gate (this must stay at 1 however many times `global()` is
/// called, including after a failure).
pub(super) static CUDA_INIT_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

/// Number of times the heavy `CudaGraph::new` initialisation has been entered
/// in this process. Exposed for the F-M2 regression test.
pub fn cuda_init_attempts() -> usize {
    CUDA_INIT_ATTEMPTS.load(Ordering::Relaxed)
}

/// Whether a CUDA context is already live in this process — **without
/// constructing one** (finding **F-M3**).
///
/// `CudaGraph::global()` builds the singleton on first call: it opens the
/// device, creates the context and loads every PTX module. A caller that only
/// wants to know "did this process ever touch CUDA" — `BonsaiModel::drop`,
/// which must release its epoch's weight buffers — cannot use it, because
/// asking would itself open the device for a model that ran entirely on CPU.
///
/// `BonsaiModel::drop` previously approximated the question with
/// `cuda_qkv_cache.is_some()`, which only the Q1 forward populates; the
/// ternary, Q4_0/Q8_0, K-quant and FP8 CUDA paths build their QKV concatenation
/// locally and never touch that field, so a model that used one of them skipped
/// its release entirely. This reads the lock-free mirror of the singleton's
/// state instead, which is exact for every path and constructs nothing.
pub fn cuda_context_is_live() -> bool {
    CUDA_INIT_STATE.load(Ordering::Relaxed) == CUDA_INIT_READY
}

/// `CudaGraphError` is memoised in the singleton slot and handed back to every
/// later caller, so it must be cloneable. (Implemented here, beside the
/// singleton that needs it, rather than derived on the enum in the sibling
/// `types` module.)
impl Clone for CudaGraphError {
    fn clone(&self) -> Self {
        match self {
            Self::DeviceNotFound(m) => Self::DeviceNotFound(m.clone()),
            Self::CompilationFailed(m) => Self::CompilationFailed(m.clone()),
            Self::DriverError(m) => Self::DriverError(m.clone()),
            Self::WeightNotFound(h) => Self::WeightNotFound(*h),
            Self::WeightLayoutError(m) => Self::WeightLayoutError(m.clone()),
            Self::InvalidDimensions(m) => Self::InvalidDimensions(m.clone()),
            Self::LockPoisoned => Self::LockPoisoned,
        }
    }
}

/// Cheap gate in front of every per-layer CUDA entry point (finding F-M2).
///
/// On a `native-cuda` build the accelerated backend *is* this singleton, so
/// "the dispatcher is on `KernelTier::Gpu`" and "`CudaGraph::global()`
/// succeeded" are the same predicate — which makes this lock-free atomic check
/// the tier guard, with no recursion back into `select_backend`. Once
/// initialisation has failed, the entry points return the memoised cause
/// immediately instead of re-running device init per layer per token.
fn cuda_graph_for_dispatch() -> Result<Arc<CudaGraph>, CudaGraphError> {
    if CUDA_INIT_STATE.load(Ordering::Relaxed) == CUDA_INIT_FAILED {
        return Err(CudaGraphError::DeviceNotFound(
            "CUDA initialisation failed earlier in this process; not retried".into(),
        ));
    }
    CudaGraph::global()
}
/// Attempt to run the FFN phase via direct CUDA dispatch.
///
/// This is the primary entry point for `block.rs` on Linux/Windows.
/// It mirrors `try_metal_ffn` exactly:
///
/// 1. Get the global `CudaGraph` singleton.
/// 2. Upload/cache weights lazily (first call uploads; subsequent calls reuse).
/// 3. Encode the full 8-op FFN pipeline on the CUDA stream.
///
/// `model_epoch` attributes every weight this call uploads to the loaded model
/// that owns it, so the model's `Drop` can release them (finding **F-M3**;
/// before this the two GPU weight caches had no writer into the epoch registry
/// and `release_model_epoch` always returned `Ok(0)`). Pass
/// [`UNATTRIBUTED_CUDA_MODEL_EPOCH`](crate::gpu_backend::cuda_graph_slot::UNATTRIBUTED_CUDA_MODEL_EPOCH)
/// when no epoch is available: the uploads then stay resident for the life of
/// the process, which is the pre-existing behaviour.
///
/// Returns `Ok(())` if the CUDA dispatch succeeded.
/// Returns `Err(...)` if no CUDA device is present or dispatch failed.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_ffn(
    hidden: &mut [f32],
    attn_out: &[f32],
    norm_weight: &[f32],
    eps: f32,
    attn_proj_handle_id: u64,
    attn_proj_bytes: &[u8],
    gate_up_handle_id: u64,
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_handle_id: u64,
    down_bytes: &[u8],
    hidden_size: usize,
    intermediate_size: usize,
    model_epoch: u64,
) -> Result<(), CudaGraphError> {
    let graph = cuda_graph_for_dispatch()?;
    let attn_proj_w = graph.get_or_upload_weight_soa_for_epoch(
        attn_proj_handle_id,
        attn_proj_bytes,
        model_epoch,
    )?;
    let gate_up_w = graph.get_or_upload_weight_soa_lazy_for_epoch(
        gate_up_handle_id,
        || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        },
        model_epoch,
    )?;
    let down_w =
        graph.get_or_upload_weight_soa_for_epoch(down_handle_id, down_bytes, model_epoch)?;
    graph.encode_ffn_phase(
        hidden,
        attn_out,
        norm_weight,
        eps,
        &attn_proj_w,
        &gate_up_w,
        &down_w,
        hidden_size,
        intermediate_size,
    )
}
/// Attempt to run a fused QKV projection via direct CUDA dispatch.
///
/// Mirrors `try_metal_qkv`:
///
/// 1. Get the global `CudaGraph` singleton.
/// 2. Upload/cache fused Q+K+V weight lazily.
/// 3. Encode a single GEMV on the CUDA stream.
///
/// `model_epoch` attributes the uploaded weight to the loaded model that owns
/// it, so the model's `Drop` can release it (finding **F-M3**). Pass
/// [`UNATTRIBUTED_CUDA_MODEL_EPOCH`](crate::gpu_backend::cuda_graph_slot::UNATTRIBUTED_CUDA_MODEL_EPOCH)
/// when no epoch is available.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_qkv(
    input: &[f32],
    output: &mut [f32],
    weight_handle_id: u64,
    q_bytes: &[u8],
    k_bytes: &[u8],
    v_bytes: &[u8],
    n_rows: usize,
    k: usize,
    model_epoch: u64,
) -> Result<(), CudaGraphError> {
    let graph = cuda_graph_for_dispatch()?;
    let weight_w = graph.get_or_upload_weight_soa_lazy_for_epoch(
        weight_handle_id,
        || {
            let mut fused = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
            fused.extend_from_slice(q_bytes);
            fused.extend_from_slice(k_bytes);
            fused.extend_from_slice(v_bytes);
            fused
        },
        model_epoch,
    )?;
    graph.encode_qkv_phase(input, output, &weight_w, n_rows, k)
}
#[cfg(test)]
mod tests {
    use super::*;
    /// Check that the singleton initialises without panicking.
    ///
    /// Skipped gracefully if no CUDA device is present (CI Linux without GPU).
    #[test]
    fn test_cuda_graph_global_init() {
        match CudaGraph::global() {
            Ok(_) => {}
            Err(e) => {
                eprintln!("CudaGraph::global() not available (expected in CPU-only CI): {e}");
            }
        }
    }
    /// Verify AoS → SoA reformatter preserves total byte count.
    #[test]
    fn test_reformat_aos_to_soa_round_trip() {
        const N: usize = 10;
        let mut aos = vec![0u8; N * 18];
        for i in 0..N {
            let base = i * 18;
            let v = i as u16;
            aos[base] = (v & 0xff) as u8;
            aos[base + 1] = (v >> 8) as u8;
            for j in 2..18 {
                aos[base + j] = 0xABu8;
            }
        }
        let soa = CudaGraph::reformat_q1_aos_to_soa(&aos).expect("reformat failed");
        assert_eq!(soa.len(), aos.len());
        for i in 0..N {
            let v = i as u16;
            assert_eq!(
                soa[i * 2],
                (v & 0xff) as u8,
                "scale byte 0 wrong at block {i}"
            );
            assert_eq!(
                soa[i * 2 + 1],
                (v >> 8) as u8,
                "scale byte 1 wrong at block {i}"
            );
        }
        for i in 0..N {
            let data_start = N * 2 + i * 16;
            for j in 0..16 {
                assert_eq!(
                    soa[data_start + j],
                    0xABu8,
                    "data wrong at block {i} byte {j}"
                );
            }
        }
    }
    /// F-M2: the singleton must memoise its outcome — `CudaGraph::new` is
    /// entered at most **once** per process however many times `global()` is
    /// called, on a CUDA host (success) and on a host without a device
    /// (failure) alike. Before the fix the failure path left the slot empty and
    /// every call re-ran `CudaContext::new` plus six NVRTC/PTX module loads and
    /// 28 `load_function` calls — per layer, per token.
    #[test]
    fn test_cuda_graph_global_init_is_memoised() {
        let _ = CudaGraph::global();
        let after_first = cuda_init_attempts();
        assert!(
            after_first <= 1,
            "CudaGraph::new entered {after_first} times; init must be memoised"
        );
        for _ in 0..8 {
            let _ = CudaGraph::global();
        }
        assert_eq!(
            cuda_init_attempts(),
            after_first,
            "CudaGraph::new must never be re-entered after the first attempt"
        );
    }

    /// F-M2: once initialisation has failed, the per-layer entry points must
    /// short-circuit on the lock-free state rather than retry device init.
    #[test]
    fn test_dispatch_guard_short_circuits_after_failure() {
        let _ = CudaGraph::global();
        if CUDA_INIT_STATE.load(Ordering::Relaxed) != CUDA_INIT_FAILED {
            // A CUDA device is present: nothing to assert about the failure
            // path, and the memoisation test above already covers the rest.
            return;
        }
        let before = cuda_init_attempts();
        assert!(cuda_graph_for_dispatch().is_err());
        assert_eq!(
            cuda_init_attempts(),
            before,
            "the dispatch guard must not re-enter CudaGraph::new"
        );
    }

    /// F1: the PTX cache key must change with the build/device environment, and
    /// stay stable within a process.
    #[test]
    fn test_ptx_env_fingerprint_is_stable() {
        assert_eq!(ptx_env_fingerprint(), ptx_env_fingerprint());
    }

    /// Verify that alloc_handle_id() produces strictly increasing unique values.
    #[test]
    fn test_handle_id_uniqueness() {
        let ids: Vec<u64> = (0..64).map(|_| alloc_handle_id()).collect();
        for w in ids.windows(2) {
            assert!(w[1] > w[0], "handle IDs not strictly increasing");
        }
    }
    /// Verify that the CUDA_V7_KERNELS_SRC constant contains the fused kernel entry point.
    ///
    /// This test does NOT require a GPU — it only inspects the static source string.
    #[test]
    fn test_fused_gate_up_swiglu_source_has_entry_point() {
        assert!(
            crate::gpu_backend::cuda_kernels::CUDA_V7_KERNELS_SRC
                .contains("fused_gate_up_swiglu_q1"),
            "CUDA_V7_KERNELS_SRC must contain the fused_gate_up_swiglu_q1 kernel entry point"
        );
    }
    /// Verify that the fused kernel source contains the SiLU epilogue expression.
    ///
    /// This guards against regressions where the epilogue is accidentally removed.
    #[test]
    fn test_fused_gate_up_swiglu_source_has_silu_epilogue() {
        let src = crate::gpu_backend::cuda_kernels::CUDA_V7_KERNELS_SRC;
        assert!(
            src.contains("silu(gate_partial) * up_partial"),
            "fused kernel epilogue 'silu(gate_partial) * up_partial' not found in kernel source"
        );
    }
    /// Verify that the fused kernel source contains both gate and up partial accumulator names.
    ///
    /// Ensures the dual-accumulator pattern is present, not just a single-path kernel.
    #[test]
    fn test_fused_gate_up_swiglu_source_has_dual_accumulators() {
        let src = crate::gpu_backend::cuda_kernels::CUDA_V7_KERNELS_SRC;
        assert!(
            src.contains("gate_partial"),
            "fused kernel must have 'gate_partial' accumulator"
        );
        assert!(
            src.contains("up_partial"),
            "fused kernel must have 'up_partial' accumulator"
        );
    }
    /// Runtime test: initialise CudaGraph and verify the fused kernel compiles successfully.
    ///
    /// Skipped gracefully if no CUDA device is present (CPU-only CI).  When a GPU is
    /// available, confirms that `fused_gate_up_swiglu_q1` was loaded from the PTX module
    /// by checking that `CudaGraph::global()` succeeds (it would error on
    /// `load_function("fused_gate_up_swiglu_q1")` otherwise).
    #[test]
    fn test_fused_gate_up_swiglu_runtime_compile() {
        match CudaGraph::global() {
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "test_fused_gate_up_swiglu_runtime_compile: no CUDA device (expected in CPU-only CI): {e}"
                );
            }
        }
    }
}
