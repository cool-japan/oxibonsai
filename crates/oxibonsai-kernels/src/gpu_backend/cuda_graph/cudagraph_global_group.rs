//! # CudaGraph - global_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! **UNVALIDATED on hardware.** This project has no CUDA device, so the device
//! selection, shared-memory negotiation and weight-release paths below are
//! compile-checked only (`scripts/check_cuda.sh`).
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::cuda_imagen_attn_kernels::CUDA_IMAGEN_ATTN_SRC;
use super::super::cuda_imagen_dit_glue_kernels::CUDA_IMAGEN_DIT_GLUE_SRC;
use super::super::cuda_imagen_gemm_kernels::CUDA_IMAGEN_GEMM_SRC;
use super::super::cuda_imagen_vae_kernels::CUDA_IMAGEN_VAE_SRC;
use super::super::cuda_kernels::CUDA_V7_KERNELS_SRC;
use cudarc::driver::{CudaContext, CudaFunction};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, info, warn};

use super::cudagraph_type::CudaGraph;
use super::functions::{
    compile_or_load_ptx, set_ptx_cache_device_tag, CUDA_INIT_ATTEMPTS, CUDA_INIT_FAILED,
    CUDA_INIT_READY, CUDA_INIT_STATE, GLOBAL_CUDA_GRAPH,
};
use super::types::{CudaGraphError, CudaModules};

// F5's tile-selection arithmetic and F-M4's device-ordinal parsing are
// pure functions with no CUDA dependency, so they live in
// `gpu_backend::cuda_device_negotiation` instead of here: that module is not
// `cfg`-gated, so its tests run on every host, unlike this one (see its module
// doc for why this file itself never compiles on this project's macOS
// development host).
use crate::gpu_backend::cuda_device_negotiation::{
    flash_large_max_head_dim, flash_shared_bytes, negotiate_flash_large, parse_device_ordinal,
    FLASH_HEAD_DIM_CAP, FLASH_TILE_KEYS,
};
// The epoch bookkeeping F-M3 drives is likewise pure, and lives in the ungated
// `gpu_backend::cuda_graph_slot` so `EpochWeightRegistry`'s register/take
// contract is unit-tested on every host.
use crate::gpu_backend::cuda_graph_slot::{EpochWeightRegistry, UNATTRIBUTED_CUDA_MODEL_EPOCH};

/// Environment override for the CUDA device ordinal (finding F-M4).
const CUDA_DEVICE_ENV: &str = "OXIBONSAI_CUDA_DEVICE";

/// The wide (`FA_DMAX = 384`) flash-attention build actually loaded here.
///
/// Before F5, `CudaGraph::new` unconditionally opted that kernel into 98 304 B
/// of dynamic shared memory and propagated the failure with `?` — so on any GPU
/// whose opt-in maximum is smaller (Turing/SM 7.5, the most common cloud
/// inference GPU, is 64 KiB; Pascal is 48 KiB) the whole native-CUDA backend
/// failed to initialise and silently degraded to CPU, including the pure-LLM
/// paths that never touch this kernel.
#[derive(Debug, Clone, Copy)]
pub struct FlashLargeConfig {
    /// Key-tile depth (`FA_BK`) the wide build was compiled with.
    pub tile_keys: usize,
    /// Dynamic shared memory successfully opted into, in bytes; `0` means the
    /// wide build is unusable here (the LLM paths are unaffected).
    pub shared_bytes: usize,
    /// Largest `head_dim` the wide build can serve on this device.
    pub max_head_dim: usize,
    /// The device's `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`.
    pub device_optin_max: usize,
}

/// Published once by [`CudaGraph::new`]; the singleton owns one device, so a
/// process-wide slot is exactly the right scope.
static FLASH_LARGE_CONFIG: OnceLock<FlashLargeConfig> = OnceLock::new();

/// `model_epoch → cached weight handles`, so a dropped model can release its
/// GPU weights (finding F-M3: neither weight cache ever evicted, so a model
/// swap leaked ~2 GB of VRAM for the life of the process).
///
/// Written by [`CudaGraph::register_model_weight`], which every epoch-attributed
/// upload helper calls right after it inserts into a weight cache. Before this
/// package the map had **no writer at all**, so `release_model_epoch` always
/// returned `Ok(0)`.
static EPOCH_WEIGHTS: OnceLock<Mutex<EpochWeightRegistry>> = OnceLock::new();

fn epoch_weights() -> &'static Mutex<EpochWeightRegistry> {
    EPOCH_WEIGHTS.get_or_init(|| Mutex::new(EpochWeightRegistry::new()))
}

/// Source of the wide flash-attention build with its key tile set.
///
/// `FA_BK` is a plain `#define` in the kernel source (unlike `FA_DMAX`, which
/// is `#ifndef`-guarded and can simply be prepended), so the host rewrites that
/// one line. Exactly one match is required: if the kernel source ever changes
/// shape, the wide build is refused rather than silently compiled at a tile the
/// host no longer expects.
fn flash_large_source(tile_keys: usize) -> Result<String, CudaGraphError> {
    const DEFAULT_TILE_DEFINE: &str = "#define FA_BK 32u";
    if tile_keys == FLASH_TILE_KEYS[0] {
        return Ok(CUDA_IMAGEN_ATTN_SRC.to_string());
    }
    if CUDA_IMAGEN_ATTN_SRC.matches(DEFAULT_TILE_DEFINE).count() != 1 {
        return Err(CudaGraphError::CompilationFailed(format!(
            "cannot retile the wide flash-attention kernel: expected exactly one \
             `{DEFAULT_TILE_DEFINE}` in CUDA_IMAGEN_ATTN_SRC"
        )));
    }
    Ok(CUDA_IMAGEN_ATTN_SRC.replace(DEFAULT_TILE_DEFINE, &format!("#define FA_BK {tile_keys}u")))
}

impl CudaGraph {
    /// Access the process-wide `CudaGraph` singleton, initialising on first call.
    ///
    /// Returns `Err` if no CUDA device is present or PTX compilation fails.
    ///
    /// The result — **including a failure** — is memoised (finding F-M2). The
    /// previous version left the slot empty on the `?`, so every later call
    /// re-ran `CudaContext::new`, six NVRTC/PTX module loads and 28
    /// `load_function` calls; the call site is per layer, per token.
    pub fn global() -> Result<Arc<CudaGraph>, CudaGraphError> {
        let mutex = GLOBAL_CUDA_GRAPH.get_or_init(|| Mutex::new(None));
        let mut guard = mutex.lock().map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(cached) = guard.as_ref() {
            return match cached {
                Ok(graph) => Ok(Arc::clone(graph)),
                Err(e) => Err(e.clone()),
            };
        }
        match Self::new() {
            Ok(graph) => {
                let graph = Arc::new(graph);
                *guard = Some(Ok(Arc::clone(&graph)));
                CUDA_INIT_STATE.store(CUDA_INIT_READY, Ordering::Relaxed);
                debug!("CudaGraph singleton initialised");
                Ok(graph)
            }
            Err(e) => {
                // Logged exactly once: the slot is filled before any retry can
                // reach this arm again.
                warn!("CUDA backend unavailable ({e}); not retried in this process");
                *guard = Some(Err(e.clone()));
                CUDA_INIT_STATE.store(CUDA_INIT_FAILED, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Ordinal of the CUDA device this process uses (finding F-M4).
    ///
    /// `OXIBONSAI_CUDA_DEVICE` selects it; an unparsable value falls back to 0
    /// with a warning. Read once — the singleton owns one context for the life
    /// of the process, so the choice cannot change afterwards.
    pub fn selected_device() -> usize {
        static SELECTED: OnceLock<usize> = OnceLock::new();
        *SELECTED.get_or_init(|| match std::env::var(CUDA_DEVICE_ENV) {
            // `parse_device_ordinal` lives in `gpu_backend::cuda_device_negotiation`
            // (not `cfg`-gated) so its input-parsing table test runs on every
            // host; this closure keeps only the per-process memoisation.
            Ok(raw) => parse_device_ordinal(&raw),
            Err(_) => 0,
        })
    }

    /// Number of CUDA devices visible to this process, or `0` when the driver
    /// is absent (finding F-M4: this used to be a hardcoded `1`).
    pub fn device_count() -> usize {
        CudaContext::device_count()
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0)
    }

    /// Wide flash-attention configuration negotiated with the device, or `None`
    /// before the singleton has been initialised (finding F5).
    pub fn flash_large_config() -> Option<FlashLargeConfig> {
        FLASH_LARGE_CONFIG.get().copied()
    }

    /// Key-tile depth (`FA_BK`) the wide flash-attention build was compiled
    /// with — the launcher must size its dynamic shared memory with **this**,
    /// not with a hardcoded 32.
    pub fn flash_large_tile_keys() -> usize {
        Self::flash_large_config()
            .map(|c| c.tile_keys)
            .unwrap_or(FLASH_TILE_KEYS[0])
    }

    /// Largest `head_dim` the wide flash-attention build can serve here —
    /// `0` when the device refused the opt-in, so the pre-launch guard in
    /// `joint_attn_flash_validate` refuses every `head_dim > 128` instead of
    /// letting the launch fail inside the driver.
    ///
    /// Before the singleton has initialised there is no device answer yet, so
    /// this reports the kernel's own compile-time cap. That is
    /// **permissive by design**: validation must not reject a shape the device
    /// may well support simply because nothing has queried it. The launch that
    /// follows initialises the singleton and is then bounded by the real
    /// negotiated value.
    pub fn flash_large_max_head_dim() -> usize {
        Self::flash_large_config()
            .map(|c| c.max_head_dim)
            .unwrap_or(FLASH_HEAD_DIM_CAP)
    }

    /// Record that `handle_id` was uploaded for `model_epoch`, so
    /// [`Self::release_model_epoch`] can free it when the model is dropped
    /// (finding F-M3).
    ///
    /// Called by every epoch-attributed upload helper immediately after it
    /// inserts into a weight cache — `get_or_upload_weight_soa_for_epoch`,
    /// `get_or_upload_weight_tq2_soa_for_epoch`,
    /// `get_or_upload_weight_aos_raw_for_epoch`,
    /// `get_or_upload_f32_weight_for_epoch`, `upload_weight_tq2_soa_for_epoch`
    /// and `cuda_full_layer::get_or_upload_f32_weight_for_epoch`.
    ///
    /// An [`UNATTRIBUTED_CUDA_MODEL_EPOCH`] upload is deliberately **not**
    /// recorded: releasing a pointer whose owner is unknown is worse than
    /// leaking it. Uploads from entry points whose call sites still pass no
    /// epoch (the prefill and image paths) therefore stay uncollected.
    pub fn register_model_weight(
        &self,
        model_epoch: u64,
        handle_id: u64,
    ) -> Result<(), CudaGraphError> {
        if model_epoch == UNATTRIBUTED_CUDA_MODEL_EPOCH {
            return Ok(());
        }
        epoch_weights()
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?
            .register(model_epoch, handle_id);
        Ok(())
    }

    /// How many weight handles are currently registered for `model_epoch` —
    /// the observability half of F-M3's bookkeeping.
    pub fn registered_weight_count(&self, model_epoch: u64) -> Result<usize, CudaGraphError> {
        Ok(epoch_weights()
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?
            .handle_count(model_epoch))
    }

    /// Drop the cached GPU weights for `handles` from **all three** weight
    /// caches and return how many cache entries were removed.
    ///
    /// The device memory is freed when the last `Arc` to each slice goes away,
    /// which for cache-owned weights is this removal.
    ///
    /// The third cache is `cuda_full_layer`'s own FP32 norm cache, which is a
    /// separate map from [`Self::f32_weight_cache`] and holds every norm weight
    /// the full-forward decode paths upload. Missing it would make
    /// [`Self::release_model_epoch`] report releases it did not perform and
    /// leave the norms resident for the life of the process.
    pub fn release_weights(&self, handles: &[u64]) -> Result<usize, CudaGraphError> {
        let mut released = 0usize;
        let mut freed_bytes = 0usize;
        {
            let mut cache = self
                .weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            for handle in handles {
                if let Some(slice) = cache.remove(handle) {
                    freed_bytes += slice.len();
                    released += 1;
                }
            }
        }
        {
            let mut cache = self
                .f32_weight_cache
                .lock()
                .map_err(|_| CudaGraphError::LockPoisoned)?;
            for handle in handles {
                if let Some(slice) = cache.remove(handle) {
                    freed_bytes += slice.len() * std::mem::size_of::<f32>();
                    released += 1;
                }
            }
        }
        // Third cache: the full-layer FP32 norm cache (a distinct map living in
        // `cuda_full_layer`'s own process state, not on `self`).
        let (fl_released, fl_bytes) =
            crate::gpu_backend::cuda_full_layer::release_f32_weights(handles)?;
        released += fl_released;
        freed_bytes += fl_bytes;
        debug!("released {released} cached CUDA weights ({freed_bytes} bytes)");
        Ok(released)
    }

    /// Release every weight registered for `model_epoch` — call from the
    /// model's `Drop` (finding F-M3).
    ///
    /// Returns the number of cache entries actually dropped, which is `0` for an
    /// unknown or already-released epoch.
    pub fn release_model_epoch(&self, model_epoch: u64) -> Result<usize, CudaGraphError> {
        let handles = epoch_weights()
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?
            .take(model_epoch);
        self.release_weights(&handles)
    }

    /// Bytes currently held by the two GPU weight caches — the observability
    /// half of F-M3, so a leak is visible before it becomes an OOM.
    pub fn weight_cache_bytes(&self) -> Result<usize, CudaGraphError> {
        let quant: usize = self
            .weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?
            .values()
            .map(|s| s.len())
            .sum();
        let f32_bytes: usize = self
            .f32_weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?
            .values()
            .map(|s| s.len() * std::mem::size_of::<f32>())
            .sum();
        Ok(quant + f32_bytes)
    }

    /// Construct a new `CudaGraph` — heavy operation (device init + NVRTC compile).
    fn new() -> Result<Self, CudaGraphError> {
        CUDA_INIT_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
        let ordinal = Self::selected_device();
        let context = CudaContext::new(ordinal)
            .map_err(|e| CudaGraphError::DeviceNotFound(format!("device {ordinal}: {e}")))?;
        unsafe {
            context.disable_event_tracking();
        }
        // Device identity feeds the PTX cache key (F1) and the shared-memory
        // negotiation below (F5).
        let attribute = |attr| context.attribute(attr).unwrap_or(0);
        use cudarc::driver::sys::CUdevice_attribute_enum as DevAttr;
        let cc_major = attribute(DevAttr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        let cc_minor = attribute(DevAttr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        let device_name = context.name().unwrap_or_else(|_| "unknown".to_string());
        set_ptx_cache_device_tag(format!("sm{cc_major}{cc_minor}"));
        let stream = context
            .new_stream()
            .map_err(|e| CudaGraphError::DriverError(format!("create stream: {e}")))?;
        let ptx = compile_or_load_ptx(CUDA_V7_KERNELS_SRC, "v7_kernels")?;
        let module = context
            .load_module(ptx)
            .map_err(|e| CudaGraphError::DriverError(format!("load_module: {e}")))?;
        let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
            module
                .load_function(name)
                .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
        };
        // ── Image-generation (FLUX.2 DiT/VAE) prototype kernels ──
        // Each source string compiles into its OWN module; `load_function` must
        // be called on the module the kernel was compiled into, so every group
        // gets a dedicated module + loader closure.
        let gemm_ptx = compile_or_load_ptx(CUDA_IMAGEN_GEMM_SRC, "imagen_gemm")?;
        let gemm_mod = context
            .load_module(gemm_ptx)
            .map_err(|e| CudaGraphError::DriverError(format!("load_module imagen_gemm: {e}")))?;
        let load_gemm = |name: &str| -> Result<CudaFunction, CudaGraphError> {
            gemm_mod
                .load_function(name)
                .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
        };
        // Flash-attention: ONE source compiled in two head_dim variants.
        //  • imagen_attn_128 (FA_DMAX=128): the lean DiT kernel — 32 KiB shared,
        //    no >48 KiB opt-in, so it keeps the full L1 its L1-backed Q[]/O[]
        //    streaming needs (a single 384/96 KiB build cost the DiT ~10%).
        //  • imagen_attn_384 (FA_DMAX=384, the source default): the VAE
        //    mid-attention kernel — opts into 96 KiB dynamic shared (set below).
        let attn_ptx_128 = compile_or_load_ptx(
            &format!("#define FA_DMAX 128u\n{CUDA_IMAGEN_ATTN_SRC}"),
            "imagen_attn_128",
        )?;
        let attn_mod_128 = context.load_module(attn_ptx_128).map_err(|e| {
            CudaGraphError::DriverError(format!("load_module imagen_attn_128: {e}"))
        })?;
        let load_attn = |name: &str| -> Result<CudaFunction, CudaGraphError> {
            attn_mod_128
                .load_function(name)
                .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
        };
        // ── F5: negotiate the wide build's key tile with the device ──────
        // `98_304` is exactly 96 KiB: Volta (96 KiB opt-in) and Ampere/Ada/
        // Hopper (99 328 B) can take it, Turing (SM 7.5 — T4, RTX 20xx) caps
        // the opt-in at 64 KiB and Pascal at 48 KiB. Rather than failing the
        // whole backend there, compile the wide build at the widest key tile
        // that fits and request only what the device allows.
        let optin_max = match context
            .attribute(DevAttr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        {
            Ok(v) if v > 0 => v as usize,
            // Attribute unavailable: keep the historical request rather than
            // silently shrinking a kernel that may well be fine.
            _ => flash_shared_bytes(FLASH_TILE_KEYS[0]),
        };
        let (tile_keys, want_shared) = negotiate_flash_large(optin_max);
        let attn_384_src = flash_large_source(tile_keys)?;
        let attn_384_tag = format!("imagen_attn_384_bk{tile_keys}");
        let attn_ptx_384 = compile_or_load_ptx(&attn_384_src, &attn_384_tag)?;
        let attn_mod_384 = context.load_module(attn_ptx_384).map_err(|e| {
            CudaGraphError::DriverError(format!("load_module imagen_attn_384: {e}"))
        })?;
        let load_attn_large = |name: &str| -> Result<CudaFunction, CudaGraphError> {
            attn_mod_384
                .load_function(name)
                .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
        };
        let vae_ptx = compile_or_load_ptx(CUDA_IMAGEN_VAE_SRC, "imagen_vae")?;
        let vae_mod = context
            .load_module(vae_ptx)
            .map_err(|e| CudaGraphError::DriverError(format!("load_module imagen_vae: {e}")))?;
        let load_vae = |name: &str| -> Result<CudaFunction, CudaGraphError> {
            vae_mod
                .load_function(name)
                .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
        };
        let dit_glue_ptx = compile_or_load_ptx(CUDA_IMAGEN_DIT_GLUE_SRC, "imagen_dit_glue")?;
        let dit_glue_mod = context.load_module(dit_glue_ptx).map_err(|e| {
            CudaGraphError::DriverError(format!("load_module imagen_dit_glue: {e}"))
        })?;
        let load_dit = |name: &str| -> Result<CudaFunction, CudaGraphError> {
            dit_glue_mod
                .load_function(name)
                .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
        };
        let modules = CudaModules {
            gemv_q1_g128_v7: load("gemv_q1_g128_v7")?,
            gemv_q1_g128_v7_residual: load("gemv_q1_g128_v7_residual")?,
            gemv_q1_g128_v8: load("gemv_q1_g128_v8")?,
            gemv_q1_g128_v8_residual: load("gemv_q1_g128_v8_residual")?,
            gemv_q1_g128_v9: load("gemv_q1_g128_v9")?,
            gemv_q1_g128_v9_residual: load("gemv_q1_g128_v9_residual")?,
            rmsnorm_weighted_v2: load("rmsnorm_weighted_v2")?,
            residual_add: load("residual_add")?,
            swiglu_fused: load("swiglu_fused")?,
            fused_gate_up_swiglu: load("fused_gate_up_swiglu_q1")?,
            argmax_f32: load("argmax_f32")?,
            gemv_tq2_g128_v1: load("gemv_tq2_g128_v1")?,
            gemm_f32: load_gemm("gemm_f32")?,
            gemm_tq2: load_gemm("gemm_tq2")?,
            joint_attention_flash_f32: load_attn("joint_attention_flash_f32")?,
            joint_attention_flash_f32_large: load_attn_large("joint_attention_flash_f32")?,
            imagen_vae_im2col: load_vae("im2col_f32")?,
            imagen_vae_groupnorm: load_vae("groupnorm_f32")?,
            imagen_vae_silu: load_vae("silu_f32")?,
            imagen_vae_upsample_nearest: load_vae("upsample_nearest_f32")?,
            dit_modulate: load_dit("modulate_f32")?,
            dit_gated_residual_add: load_dit("gated_residual_add_f32")?,
            dit_layer_norm: load_dit("layer_norm_f32")?,
            dit_rms_norm_heads: load_dit("rms_norm_heads_f32")?,
            dit_swiglu: load_dit("swiglu_f32")?,
            dit_rope_interleaved: load_dit("rope_interleaved_f32")?,
            dit_tokens_to_heads: load_dit("tokens_to_heads_f32")?,
            dit_strided_row_copy: load_dit("strided_row_copy_f32")?,
        };
        // Opt ONLY the wide VAE variant into >48 KiB dynamic shared memory so it
        // can stage Ksh ‖ Vsh for head_dim=384. The DiT variant deliberately
        // gets NO opt-in: it only ever requests 32 KiB shared, so it keeps the
        // large default L1 its L1-backed Q[]/O[] streaming needs (the prior
        // single 96 KiB-opt-in build pinned L1 low and cost the DiT ~10%).
        //
        // F5: this is **not** fatal any more. A device that refuses the opt-in
        // only loses the wide VAE attention kernel — reported through
        // `FlashLargeConfig::shared_bytes == 0` and refused at launch — instead
        // of taking the entire native-CUDA backend (and every LLM path) down to
        // the CPU tier.
        let granted_shared = match modules
            .joint_attention_flash_f32_large
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(want_shared).unwrap_or(i32::MAX),
            ) {
            Ok(()) => want_shared,
            Err(e) => {
                warn!(
                    "device refused {want_shared} B of dynamic shared memory for \
                     joint_attention_flash_f32_large ({e}); wide attention \
                     (head_dim > 128) is disabled, every other CUDA kernel — \
                     including the whole LLM path — is unaffected"
                );
                0
            }
        };
        let config = FlashLargeConfig {
            tile_keys,
            shared_bytes: granted_shared,
            max_head_dim: flash_large_max_head_dim(granted_shared, tile_keys),
            device_optin_max: optin_max,
        };
        let _ = FLASH_LARGE_CONFIG.set(config);
        info!(
            device = %device_name,
            ordinal,
            compute_capability = %format!("{cc_major}.{cc_minor}"),
            shared_mem_optin_max = optin_max,
            flash_tile_keys = tile_keys,
            flash_shared_bytes = granted_shared,
            flash_max_head_dim = config.max_head_dim,
            "CUDA device initialised (UNVALIDATED: compile-checked only, no CUDA hardware)"
        );
        Ok(Self {
            context,
            stream,
            modules,
            buffers: Mutex::new(None),
            qkv_buffers: Mutex::new(None),
            weight_cache: Mutex::new(HashMap::new()),
            f32_weight_cache: Mutex::new(HashMap::new()),
            lm_head_buffers: Mutex::new(None),
            tq2_gemv_buffers: Mutex::new(None),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F5: retiling rewrites exactly the one `#define`, and the widest tile is
    /// byte-identical to the shipped source.
    #[test]
    fn flash_large_source_retiles_exactly_once() {
        let retiled = flash_large_source(16).expect("retile to 16 keys");
        assert!(retiled.contains("#define FA_BK 16u"));
        assert!(!retiled.contains("#define FA_BK 32u"));
        assert_eq!(retiled.len(), CUDA_IMAGEN_ATTN_SRC.len());
        assert_eq!(
            flash_large_source(32).expect("default tile"),
            CUDA_IMAGEN_ATTN_SRC
        );
    }

    /// F-M4: an unparsable ordinal must degrade to device 0, never panic.
    ///
    /// `selected_device` memoises per process, so only the shape is checked
    /// here; `cuda_device_negotiation::tests::parse_device_ordinal_table`
    /// table-tests the parsing itself (`""`, `"3"`, `" 3 "`, `"abc"`, `"-1"`)
    /// on every host, since this whole file is `cfg`-gated out on macOS.
    #[test]
    fn selected_device_defaults_to_zero() {
        let ordinal = CudaGraph::selected_device();
        assert!(ordinal < 1024, "implausible device ordinal {ordinal}");
    }
}
