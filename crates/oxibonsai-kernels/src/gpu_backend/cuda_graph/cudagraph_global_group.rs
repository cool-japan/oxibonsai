//! # CudaGraph - global_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::cuda_imagen_attn_kernels::CUDA_IMAGEN_ATTN_SRC;
use super::super::cuda_imagen_dit_glue_kernels::CUDA_IMAGEN_DIT_GLUE_SRC;
use super::super::cuda_imagen_gemm_kernels::CUDA_IMAGEN_GEMM_SRC;
use super::super::cuda_imagen_vae_kernels::CUDA_IMAGEN_VAE_SRC;
use super::super::cuda_kernels::CUDA_V7_KERNELS_SRC;
use cudarc::driver::{CudaContext, CudaFunction};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::debug;

use super::cudagraph_type::CudaGraph;
use super::functions::{compile_or_load_ptx, GLOBAL_CUDA_GRAPH};
use super::types::{CudaGraphError, CudaModules};

impl CudaGraph {
    /// Access the process-wide `CudaGraph` singleton, initialising on first call.
    ///
    /// Returns `Err` if no CUDA device is present or PTX compilation fails.
    pub fn global() -> Result<Arc<CudaGraph>, CudaGraphError> {
        let mutex = GLOBAL_CUDA_GRAPH.get_or_init(|| Mutex::new(None));
        let mut guard = mutex.lock().map_err(|_| CudaGraphError::LockPoisoned)?;
        if let Some(ref cached) = *guard {
            return Ok(Arc::clone(cached));
        }
        let graph = Arc::new(Self::new()?);
        *guard = Some(Arc::clone(&graph));
        debug!("CudaGraph singleton initialised");
        Ok(graph)
    }
    /// Construct a new `CudaGraph` — heavy operation (device init + NVRTC compile).
    fn new() -> Result<Self, CudaGraphError> {
        let context =
            CudaContext::new(0).map_err(|e| CudaGraphError::DeviceNotFound(format!("{e}")))?;
        unsafe {
            context.disable_event_tracking();
        }
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
        let attn_ptx_384 = compile_or_load_ptx(CUDA_IMAGEN_ATTN_SRC, "imagen_attn_384")?;
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
        // can stage Ksh ‖ Vsh for head_dim=384 (96 KiB). Max = FA_BK(32) *
        // DIT_FLASH_HEAD_DIM_CAP(384) * 2 * sizeof(f32) = 98304 bytes, within the
        // Ampere 100 KiB per-block limit. The DiT variant deliberately gets NO
        // opt-in: it only ever requests 32 KiB shared, so it keeps the large
        // default L1 its L1-backed Q[]/O[] streaming needs (the prior single
        // 96 KiB-opt-in build pinned L1 low and cost the DiT ~10%). Failing here
        // would only surface as a launch error for the VAE case, so it is fatal.
        modules
            .joint_attention_flash_f32_large
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                98_304,
            )
            .map_err(|e| {
                CudaGraphError::DriverError(format!(
                    "set max dynamic shared mem on joint_attention_flash_f32_large: {e}"
                ))
            })?;
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
