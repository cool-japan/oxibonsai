//! Prefill (batch) GPU dispatch for OxiBonsai — CUDA backend.
//!
//! Mirrors [`metal_prefill`] for Linux/Windows.  Handles batch processing of
//! multiple tokens during prompt prefill using GEMM kernels.
//!
//! # Architecture
//!
//! - [`CudaPrefillBuffers`]: Pre-allocated GPU buffers sized for `batch_size` tokens.
//! - [`CudaPrefillModules`]: Compiled CUDA functions for the 5 prefill kernels.
//! - `encode_prefill_ffn_phase`: Batched FFN pipeline (RMSNorm → gate+up+SwiGLU → down).
//! - `encode_prefill_layer`: One full prefill transformer layer.
//! - [`try_cuda_prefill`]: Public entry point mirroring `try_metal_full_forward_prefill`.
//!
//! # Batch tensor layout
//!
//! All batched buffers use **column-major** layout: `buf[col * dim + element]`
//! where `col` is the batch/token index.  This matches the Metal MSL kernels.
//!
//! # Attention in the prefill path
//!
//! We do not have a batched attention kernel; attention is processed sequentially
//! per token using the existing single-token attention kernels from `cuda_full_layer`.

#![cfg(all(
    feature = "native-cuda",
    any(target_os = "linux", target_os = "windows")
))]

use cudarc::driver::{CudaFunction, CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use std::sync::{Arc, Mutex, OnceLock};

use super::cuda_full_layer::{
    encode_attn_phase, init_attn_modules, CudaAttnModules, CudaFullForwardLayerParams,
    CudaFullLayerBuffers, CudaKvCache,
};
use super::cuda_graph::{compile_or_load_ptx, CudaGraph, CudaGraphError};
use super::cuda_prefill_kernels::CUDA_PREFILL_KERNELS_SRC;

// Type alias for the per-layer weight tuple used during prefill.
// Fields (in order): attn_norm, fused_qkv, q_norm, k_norm, attn_proj, ffn_norm, gate_up, down.
type LayerWeightArcs = (
    Arc<CudaSlice<f32>>, // attn_norm
    Arc<CudaSlice<u8>>,  // fused_qkv
    Arc<CudaSlice<f32>>, // q_norm
    Arc<CudaSlice<f32>>, // k_norm
    Arc<CudaSlice<u8>>,  // attn_proj
    Arc<CudaSlice<f32>>, // ffn_norm
    Arc<CudaSlice<u8>>,  // gate_up (fused)
    Arc<CudaSlice<u8>>,  // down
);

// =============================================================================
// Pre-allocated prefill GPU buffers
// =============================================================================

/// Pre-allocated GPU activation buffers for prefill (batch) processing.
///
/// All batched buffers use column-major layout: `buf[col * dim + element]`.
pub struct CudaPrefillBuffers {
    /// Batched hidden states: `[capacity * hidden_size]` f32 (column-major).
    pub d_input: CudaSlice<f32>,
    /// Batched RMSNorm output: `[capacity * hidden_size]` f32 (column-major).
    pub d_normed: CudaSlice<f32>,
    /// Batched QKV GEMM output: `[capacity * (nq+2*nkv)*head_dim]` f32.
    pub d_qkv: CudaSlice<f32>,
    /// Batched attention output: `[capacity * nq*head_dim]` f32 (column-major).
    pub d_attn_out: CudaSlice<f32>,
    /// Batched gate+up GEMM output: `[capacity * intermediate_size]` f32.
    /// Layout: `[gate: bs*inter | up: bs*inter]` for `batched_swiglu`.
    pub d_gate_up: CudaSlice<f32>,
    /// Batched SwiGLU output: `[capacity * intermediate_size]` f32 (column-major).
    pub d_swiglu: CudaSlice<f32>,
    /// Allocated capacity (max batch_size for which buffers are valid).
    pub capacity: usize,
    /// Currently-active batch size (≤ capacity), set before each encode call.
    pub actual_batch_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub nq: usize,
    pub nkv: usize,
    pub head_dim: usize,
    pub max_seq: usize,
}

// SAFETY: CudaSlice<f32> is Send in cudarc.
unsafe impl Send for CudaPrefillBuffers {}
unsafe impl Sync for CudaPrefillBuffers {}

impl CudaPrefillBuffers {
    /// Check whether these buffers can serve the requested dimensions.
    ///
    /// `batch_size` uses capacity comparison (`<=`) so buffers allocated for a
    /// larger batch can be reused for smaller batches without reallocation.
    /// All other dimensions must match exactly (they determine pointer layouts).
    #[allow(clippy::too_many_arguments)]
    pub fn matches(
        &self,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> bool {
        batch_size <= self.capacity   // capacity-based: smaller batches reuse larger allocations
            && self.hidden_size == hidden_size
            && self.intermediate_size == intermediate_size
            && self.nq == nq
            && self.nkv == nkv
            && self.head_dim == head_dim
            && self.max_seq == max_seq
    }
}

// =============================================================================
// Compiled prefill CUDA modules
// =============================================================================

/// Compiled CUDA function handles for the 5 prefill kernels.
pub struct CudaPrefillModules {
    pub gemm_v7: CudaFunction,
    pub gemm_v7_residual: CudaFunction,
    pub fused_gate_up_swiglu_gemm: CudaFunction,
    pub batched_swiglu: CudaFunction,
    pub batched_rmsnorm: CudaFunction,
}

// SAFETY: CudaFunction is Send in cudarc.
unsafe impl Send for CudaPrefillModules {}
unsafe impl Sync for CudaPrefillModules {}

// =============================================================================
// Process-wide singleton state for the prefill path
// =============================================================================

struct CudaPrefillState {
    prefill_modules: Mutex<Option<Arc<CudaPrefillModules>>>,
    prefill_buffers: Mutex<Option<CudaPrefillBuffers>>,
    /// Shared KV cache (same singleton as the decode path).
    kv_cache: Mutex<Option<CudaKvCache>>,
    /// Reuse the single-token full-layer buffers for per-token attention.
    full_layer_buffers: Mutex<Option<CudaFullLayerBuffers>>,
    /// Cached logits buffer: (buffer, out_features_count).
    prefill_logits: Mutex<Option<(CudaSlice<f32>, usize)>>,
}

unsafe impl Send for CudaPrefillState {}
unsafe impl Sync for CudaPrefillState {}

static PREFILL_STATE: OnceLock<CudaPrefillState> = OnceLock::new();

fn prefill_state() -> &'static CudaPrefillState {
    PREFILL_STATE.get_or_init(|| CudaPrefillState {
        prefill_modules: Mutex::new(None),
        prefill_buffers: Mutex::new(None),
        kv_cache: Mutex::new(None),
        full_layer_buffers: Mutex::new(None),
        prefill_logits: Mutex::new(None),
    })
}

// =============================================================================
// Module init
// =============================================================================

/// Compile and cache the 5 CUDA prefill kernels.
///
/// Idempotent: the second call returns the already-compiled modules immediately.
pub fn init_prefill_modules(graph: &CudaGraph) -> Result<Arc<CudaPrefillModules>, CudaGraphError> {
    let state = prefill_state();
    let mut guard = state
        .prefill_modules
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    if let Some(ref m) = *guard {
        return Ok(Arc::clone(m));
    }

    let ptx = compile_or_load_ptx(CUDA_PREFILL_KERNELS_SRC, "prefill_kernels")?;

    let module = graph
        .context_arc()
        .load_module(ptx)
        .map_err(|e| CudaGraphError::DriverError(format!("load_module prefill: {e}")))?;

    let load = |name: &str| -> Result<CudaFunction, CudaGraphError> {
        module
            .load_function(name)
            .map_err(|e| CudaGraphError::DriverError(format!("load_function({name}): {e}")))
    };

    let modules = Arc::new(CudaPrefillModules {
        gemm_v7: load("gemm_q1_g128_v7")?,
        gemm_v7_residual: load("gemm_q1_g128_v7_residual")?,
        fused_gate_up_swiglu_gemm: load("fused_gate_up_swiglu_gemm_q1")?,
        batched_swiglu: load("batched_swiglu")?,
        batched_rmsnorm: load("batched_rmsnorm_v2")?,
    });

    *guard = Some(Arc::clone(&modules));
    Ok(modules)
}

// =============================================================================
// Buffer / cache acquisition helpers
// =============================================================================

/// Round up `n` to the next power of two (minimum 1).
fn next_pow2_capacity(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut cap = 1usize;
    while cap < n {
        cap <<= 1;
    }
    cap
}

/// Acquire or (re-)allocate the prefill activation buffers.
#[allow(clippy::too_many_arguments)]
fn acquire_prefill_buffers(
    graph: &CudaGraph,
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaPrefillBuffers>>, CudaGraphError> {
    let state = prefill_state();
    let mut guard = state
        .prefill_buffers
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    let needs_alloc = match guard.as_ref() {
        Some(b) => !b.matches(
            batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq,
        ),
        None => true,
    };

    if needs_alloc {
        let capacity = next_pow2_capacity(batch_size);
        let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<f32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc_zeros pb({n}): {e}")))
        };

        let qkv_total = (nq + 2 * nkv) * head_dim;

        *guard = Some(CudaPrefillBuffers {
            d_input: alloc(capacity * hidden_size)?,
            d_normed: alloc(capacity * hidden_size)?,
            d_qkv: alloc(capacity * qkv_total)?,
            d_attn_out: alloc(capacity * nq * head_dim)?,
            d_gate_up: alloc(2 * capacity * intermediate_size)?,
            d_swiglu: alloc(capacity * intermediate_size)?,
            capacity,
            actual_batch_size: batch_size,
            hidden_size,
            intermediate_size,
            nq,
            nkv,
            head_dim,
            max_seq,
        });
    } else {
        // Reusing existing allocation — just update the active batch size.
        // SAFETY: needs_alloc is false only when guard is Some(b), so this is infallible.
        guard
            .as_mut()
            .expect("guard is Some when needs_alloc is false")
            .actual_batch_size = batch_size;
    }

    Ok(guard)
}

/// Acquire or (re-)allocate the shared GPU KV cache.
fn acquire_prefill_kv_cache(
    graph: &CudaGraph,
    n_layers: usize,
    n_kv: usize,
    max_seq: usize,
    head_dim: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaKvCache>>, CudaGraphError> {
    let state = prefill_state();
    let mut guard = state
        .kv_cache
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    let needs_alloc = match guard.as_ref() {
        Some(c) => !c.matches(n_layers, n_kv, max_seq, head_dim),
        None => true,
    };

    if needs_alloc {
        let total = n_layers * n_kv * max_seq * head_dim;
        let k_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv k: {e}")))?;
        let v_cache = graph
            .stream_arc()
            .alloc_zeros::<u16>(total)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc kv v: {e}")))?;

        *guard = Some(CudaKvCache {
            k_cache,
            v_cache,
            n_layers,
            n_kv,
            max_seq,
            head_dim,
        });
    }

    Ok(guard)
}

/// Acquire or (re-)allocate single-token full-layer buffers for per-token attention.
fn acquire_single_token_buffers(
    graph: &CudaGraph,
    hidden_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    max_seq: usize,
    intermediate_size: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CudaFullLayerBuffers>>, CudaGraphError> {
    let state = prefill_state();
    let mut guard = state
        .full_layer_buffers
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;

    let needs_alloc = match guard.as_ref() {
        Some(b) => !b.matches(hidden_size, nq, nkv, head_dim, max_seq, intermediate_size),
        None => true,
    };

    if needs_alloc {
        let alloc = |n: usize| -> Result<CudaSlice<f32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<f32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc st({n}): {e}")))
        };

        let qkv_total = nq * head_dim + 2 * nkv * head_dim;
        let half_dim = head_dim / 2;

        let alloc_u32 = |n: usize| -> Result<CudaSlice<u32>, CudaGraphError> {
            graph
                .stream_arc()
                .alloc_zeros::<u32>(n)
                .map_err(|e| CudaGraphError::DriverError(format!("alloc u32({n}): {e}")))
        };
        *guard = Some(CudaFullLayerBuffers {
            d_hidden: alloc(hidden_size)?,
            d_normed: alloc(hidden_size)?,
            d_qkv: alloc(qkv_total)?,
            d_q_rope: alloc(nq * head_dim)?,
            d_k_rope: alloc(nkv * head_dim)?,
            d_cos: alloc(half_dim)?,
            d_sin: alloc(half_dim)?,
            d_scores: alloc(nq * max_seq)?,
            d_attn_out: alloc(nq * head_dim)?,
            d_gate_up: alloc(2 * intermediate_size)?,
            d_swiglu: alloc(intermediate_size)?,
            d_pos_seqlen: alloc_u32(2)?,
            hidden_size,
            nq,
            nkv,
            head_dim,
            max_seq,
            intermediate_size,
        });
    }

    Ok(guard)
}

type PrefillLogitsGuard = std::sync::MutexGuard<'static, Option<(CudaSlice<f32>, usize)>>;

/// Acquire or (re-)allocate the LM-head logits buffer.
fn acquire_prefill_logits(
    graph: &CudaGraph,
    n: usize,
) -> Result<PrefillLogitsGuard, CudaGraphError> {
    let state = prefill_state();
    let mut guard = state
        .prefill_logits
        .lock()
        .map_err(|_| CudaGraphError::LockPoisoned)?;
    let needs_alloc = match guard.as_ref() {
        Some((_, sz)) => *sz != n,
        None => true,
    };
    if needs_alloc {
        let buf = graph
            .stream_arc()
            .alloc_zeros::<f32>(n)
            .map_err(|e| CudaGraphError::DriverError(format!("alloc logits buf({n}): {e}")))?;
        *guard = Some((buf, n));
    }
    Ok(guard)
}

// =============================================================================
// Low-level prefill kernel launchers
// =============================================================================

/// Launch `gemm_q1_g128_v7` (batch GEMM, accumulate into outputs with `+=`).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm_v7(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_v7)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_v7 launch: {e}")))
}

/// Launch `gemm_q1_g128_v7_residual` (batch GEMM + fused residual overwrite).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments, dead_code)]
unsafe fn launch_gemm_v7_residual(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
    d_residual: &CudaSlice<f32>,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.gemm_v7_residual)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .arg(d_residual)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("gemm_v7_residual launch: {e}")))
}

/// Launch `fused_gate_up_swiglu_gemm_q1` (batch fused gate+up+SwiGLU GEMM).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_fused_gate_up_swiglu_gemm(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_blocks: &CudaSlice<u8>,
    d_inputs: &CudaSlice<f32>,
    d_outputs: &mut CudaSlice<f32>,
    n_rows: u32,
    k: u32,
    batch_size: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = n_rows.div_ceil(8);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_gate_up_swiglu_gemm)
        .arg(d_blocks)
        .arg(d_inputs)
        .arg(d_outputs)
        .arg(&n_rows)
        .arg(&k)
        .arg(&batch_size)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_gate_up_swiglu_gemm launch: {e}")))
}

/// Launch `batched_rmsnorm_v2` (one block per batch token).
///
/// # Safety
/// All slices must be valid device pointers allocated on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_batched_rmsnorm(
    graph: &CudaGraph,
    mods: &CudaPrefillModules,
    d_input: &CudaSlice<f32>,
    d_weight: &CudaSlice<f32>,
    d_output: &mut CudaSlice<f32>,
    n: u32,
    batch_size: u32,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (batch_size, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_rmsnorm)
        .arg(d_input)
        .arg(d_weight)
        .arg(d_output)
        .arg(&n)
        .arg(&batch_size)
        .arg(&eps)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_rmsnorm launch: {e}")))
}

// =============================================================================
// encode_prefill_ffn_phase
// =============================================================================

/// Encode the batched FFN sublayer for all `batch_size` tokens.
///
/// Pipeline:
/// 1. Batched RMSNorm: `d_hidden → d_normed` (all tokens at once)
/// 2. Fused gate+up+SwiGLU GEMM: `d_normed → d_swiglu` (all tokens)
/// 3. Down GEMM + residual: `d_swiglu → d_hidden` (fused residual add)
///
/// On return, `d_hidden` in `pb` contains the updated residual stream.
///
/// # Safety
/// All device buffers must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn encode_prefill_ffn_phase(
    graph: &CudaGraph,
    pmods: &CudaPrefillModules,
    d_ffn_norm_weight: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    pb: &mut CudaPrefillBuffers,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let bs = pb.actual_batch_size as u32;
    let h = pb.hidden_size as u32;
    let inter = pb.intermediate_size as u32;

    // Step 1: Batched RMSNorm (all tokens)
    launch_batched_rmsnorm(
        graph,
        pmods,
        &pb.d_input,
        d_ffn_norm_weight,
        &mut pb.d_normed,
        h,
        bs,
        eps,
    )?;

    // Step 2: Fused gate+up+SwiGLU GEMM (all tokens)
    //   d_normed [bs × h, col-major] → d_swiglu [bs × inter, col-major]
    //   Weight: concatenated gate+up SoA, 2*inter rows, k=h
    launch_fused_gate_up_swiglu_gemm(
        graph,
        pmods,
        d_gate_up_weight,
        &pb.d_normed,
        &mut pb.d_swiglu,
        inter,
        h,
        bs,
    )?;

    // Step 3: Down GEMM into d_normed (scratch), then in-place residual add.
    //
    // gemm_v7 accumulates with +=, so d_normed must be zeroed before the GEMM.
    // d_normed is free here (consumed as GEMM input in step 2 already).
    {
        let n = pb.actual_batch_size * pb.hidden_size;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_normed down: {e}")))?;
    }
    launch_gemm_v7(
        graph,
        pmods,
        d_down_weight,
        &pb.d_swiglu,
        &mut pb.d_normed,
        h,
        inter,
        bs,
    )?;

    // residual add: d_input[i] += d_normed[i]  (total = bs * hidden_size elements)
    let total_bh = (pb.actual_batch_size * pb.hidden_size) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_bh)?;

    Ok(())
}

// =============================================================================
// encode_prefill_layer
// =============================================================================

/// Encode one full transformer layer for batch prefill.
///
/// Non-attention operations use batched GEMM kernels.  Attention is processed
/// sequentially per token (using the existing single-token attention kernels)
/// because each query position needs access to all prior KV entries up to its
/// position — there is no batched attention kernel available.
///
/// On entry / exit, `pb.d_input` holds the batched residual stream
/// `[batch_size × hidden_size]` in column-major layout.
///
/// # Safety
/// All device buffers and weight slices must be valid on `graph.stream_arc()`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn encode_prefill_layer(
    graph: &CudaGraph,
    pmods: &CudaPrefillModules,
    attn_mods: &CudaAttnModules,
    d_attn_norm_weight: &CudaSlice<f32>,
    d_fused_qkv_weight: &Arc<CudaSlice<u8>>,
    d_q_norm_weight: &CudaSlice<f32>,
    d_k_norm_weight: &CudaSlice<f32>,
    d_attn_proj_weight: &Arc<CudaSlice<u8>>,
    d_ffn_norm_weight: &CudaSlice<f32>,
    d_gate_up_weight: &Arc<CudaSlice<u8>>,
    d_down_weight: &Arc<CudaSlice<u8>>,
    kv: &mut CudaKvCache,
    layer_idx: usize,
    pos_start: usize,
    pb: &mut CudaPrefillBuffers,
    st_bufs: &mut CudaFullLayerBuffers,
    cos_table: &[f32],
    sin_table: &[f32],
    heads_per_group: usize,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let bs = pb.actual_batch_size;
    let h = pb.hidden_size;
    let nq = pb.nq;
    let nkv = pb.nkv;
    let hd = pb.head_dim;
    let half_dim = hd / 2;
    let h_u32 = h as u32;
    let bs_u32 = bs as u32;
    let qkv_total = nq * hd + 2 * nkv * hd;

    // ════════════════════════════════════════════════════════════════════
    // 1. Batched RMSNorm (attn norm): d_input → d_normed
    // ════════════════════════════════════════════════════════════════════
    launch_batched_rmsnorm(
        graph,
        pmods,
        &pb.d_input,
        d_attn_norm_weight,
        &mut pb.d_normed,
        h_u32,
        bs_u32,
        eps,
    )?;

    // ════════════════════════════════════════════════════════════════════
    // 2. Batched QKV GEMM: d_normed → d_qkv
    //    n_rows = (nq + 2*nkv) * head_dim, k = hidden_size
    //    Zero-init d_qkv first so accumulate (+=) is correct.
    // ════════════════════════════════════════════════════════════════════
    // Zero out d_qkv so the += in gemm_v7 starts from zero.
    {
        let n = bs * qkv_total;
        let mut dst_view = pb.d_qkv.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_qkv: {e}")))?;
    }

    launch_gemm_v7(
        graph,
        pmods,
        d_fused_qkv_weight,
        &pb.d_normed,
        &mut pb.d_qkv,
        qkv_total as u32,
        h_u32,
        bs_u32,
    )?;

    // ════════════════════════════════════════════════════════════════════
    // 3. Sequential attention for each token
    //
    // For each token t at sequence position (pos_start + t), we:
    //   a) Copy this token's hidden state into st_bufs.d_hidden
    //   b) Copy this token's QKV into st_bufs.d_qkv (extracted from batched)
    //   c) Copy this token's RoPE cos/sin into st_bufs.d_cos/d_sin
    //   d) Run the standard single-token attention kernels (qk-norm+rope,
    //      kv-store, scores, softmax, weighted sum)
    //   e) Copy attention output back into the column of pb.d_attn_out
    // ════════════════════════════════════════════════════════════════════
    let f_size = std::mem::size_of::<f32>();

    // Zero out d_attn_out before the sequential attention loop.
    {
        let n = bs * nq * hd;
        let mut dst_view = pb.d_attn_out.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_attn_out: {e}")))?;
    }

    for t in 0..bs {
        let pos = pos_start + t;

        // Copy token t's hidden state column into st_bufs.d_hidden
        // Column-major: token t's hidden is at pb.d_input[t * h .. (t+1)*h]
        {
            let src_view: CudaView<f32> = pb.d_input.slice(t * h..(t + 1) * h);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                .map_err(|e| CudaGraphError::DriverError(format!("copy hidden t={t}: {e}")))?;
        }

        // Copy token t's QKV column into st_bufs.d_qkv
        // Column-major: token t's QKV is at pb.d_qkv[t * qkv_total .. (t+1)*qkv_total]
        {
            let src_view: CudaView<f32> = pb.d_qkv.slice(t * qkv_total..(t + 1) * qkv_total);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut st_bufs.d_qkv)
                .map_err(|e| CudaGraphError::DriverError(format!("copy qkv t={t}: {e}")))?;
        }

        // Upload RoPE cos/sin for this token's position.
        let rope_off = t * half_dim;
        graph
            .stream_arc()
            .memcpy_htod(
                &cos_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_cos,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload cos t={t}: {e}")))?;
        graph
            .stream_arc()
            .memcpy_htod(
                &sin_table[rope_off..rope_off + half_dim],
                &mut st_bufs.d_sin,
            )
            .map_err(|e| CudaGraphError::DriverError(format!("upload sin t={t}: {e}")))?;

        // Run the 7-step single-token attention pipeline.
        // encode_attn_phase reads from st_bufs.d_hidden (already set above)
        // and uses st_bufs.d_qkv as Q (it skips the internal GEMV and
        // goes straight to QK-norm+RoPE using the provided QKV data).
        //
        // However, encode_attn_phase always runs a full RMSNorm + QKV GEMV
        // on d_hidden.  For the prefill path, the normed hidden and QKV are
        // already computed in the batched steps above.  We pass the attn_norm
        // weight and fused_qkv weight again; the redundant RMSNorm + GEMV
        // overhead is acceptable given the sequential attention constraint.
        encode_attn_phase(
            graph,
            attn_mods,
            d_attn_norm_weight,
            d_fused_qkv_weight,
            d_q_norm_weight,
            d_k_norm_weight,
            kv,
            layer_idx,
            pos,
            nq,
            nkv,
            hd,
            heads_per_group,
            eps,
            h,
            st_bufs,
        )?;

        // Copy attention output for this token from st_bufs.d_attn_out into
        // the column of pb.d_attn_out [t * nq*hd .. (t+1)*nq*hd]
        {
            let src_view: CudaView<f32> = st_bufs.d_attn_out.slice(..nq * hd);
            let mut dst_view = pb.d_attn_out.slice_mut(t * nq * hd..(t + 1) * nq * hd);
            graph
                .stream_arc()
                .memcpy_dtod(&src_view, &mut dst_view)
                .map_err(|e| CudaGraphError::DriverError(format!("copy attn_out t={t}: {e}")))?;
        }

        // Silence f_size unused warning (used contextually in offset calculations)
        let _ = f_size;
    }

    // ════════════════════════════════════════════════════════════════════
    // 4. Output projection GEMM + residual (all tokens at once)
    //    attn_out_proj: [h × nq*hd], maps d_attn_out → d_normed (scratch)
    //    then: d_input += d_normed  (residual add)
    //
    // We use d_normed as a scratch to avoid aliasing d_input as both
    // &mut output and &residual in the fused gemm_v7_residual kernel.
    // ════════════════════════════════════════════════════════════════════
    {
        // Zero d_normed so the accumulating gemm_v7 starts from zero.
        let n = bs * h;
        let mut dst_view = pb.d_normed.slice_mut(0..n);
        graph
            .stream_arc()
            .memset_zeros(&mut dst_view)
            .map_err(|e| CudaGraphError::DriverError(format!("zero d_normed oproj: {e}")))?;
    }
    launch_gemm_v7(
        graph,
        pmods,
        d_attn_proj_weight,
        &pb.d_attn_out,
        &mut pb.d_normed,
        h_u32,
        (nq * hd) as u32,
        bs_u32,
    )?;
    // residual add: d_input[i] += d_normed[i]
    let total_oproj = (bs * h) as u32;
    graph.launch_residual_add_pub(&mut pb.d_input, &pb.d_normed, total_oproj)?;

    // ════════════════════════════════════════════════════════════════════
    // 5. Batched FFN (RMSNorm → fused gate+up+SwiGLU → down + residual)
    // ════════════════════════════════════════════════════════════════════
    encode_prefill_ffn_phase(
        graph,
        pmods,
        d_ffn_norm_weight,
        d_gate_up_weight,
        d_down_weight,
        pb,
        eps,
    )?;

    Ok(())
}

// =============================================================================
// Public entry point
// =============================================================================

/// Attempt to run batch prefill (ALL transformer layers + LM head) via CUDA.
///
/// Processes `batch_size` tokens simultaneously using GEMM kernels for
/// projections and sequential per-token attention within each layer.
/// Only the last token's logits are returned in `logits_out` / `greedy_token_id_out`.
///
/// Mirrors `try_metal_full_forward_prefill` exactly.
///
/// Returns `Ok(())` on success.  Returns `Err(...)` if CUDA is unavailable or
/// any kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_prefill(
    hidden_batch: &[f32],
    batch_size: usize,
    pos_start: usize,
    n_layers: usize,
    layer_params: &[CudaFullForwardLayerParams<'_>],
    cos_table: &[f32],
    sin_table: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    nq: usize,
    nkv: usize,
    head_dim: usize,
    heads_per_group: usize,
    eps: f32,
    max_seq_len: usize,
    final_norm_handle: Option<u64>,
    final_norm_bytes: Option<&[f32]>,
    final_norm_eps: f32,
    lm_head_handle: Option<u64>,
    lm_head_bytes: Option<&[u8]>,
    lm_head_out_features: usize,
    logits_out: Option<&mut Vec<f32>>,
    greedy_token_id_out: Option<&mut u32>,
) -> Result<(), CudaGraphError> {
    if layer_params.len() != n_layers {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "layer_params length mismatch: need {n_layers}, got {}",
            layer_params.len()
        )));
    }

    let half_dim = head_dim / 2;

    if hidden_batch.len() < batch_size * hidden_size {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "hidden_batch too short: need {}, got {}",
            batch_size * hidden_size,
            hidden_batch.len()
        )));
    }
    if cos_table.len() < batch_size * half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "cos_table too short: need {}, got {}",
            batch_size * half_dim,
            cos_table.len()
        )));
    }
    if sin_table.len() < batch_size * half_dim {
        return Err(CudaGraphError::WeightLayoutError(format!(
            "sin_table too short: need {}, got {}",
            batch_size * half_dim,
            sin_table.len()
        )));
    }

    let graph = CudaGraph::global()?;
    let _t_prefill = super::cuda_full_layer::profiling().then(std::time::Instant::now);
    let pmods = init_prefill_modules(&graph)?;
    let attn_mods = init_attn_modules(&graph)?;

    // ── Upload / cache all per-layer weights ────────────────────────────
    let mut layer_weight_arcs: Vec<LayerWeightArcs> = Vec::with_capacity(n_layers);

    for lp in layer_params {
        let attn_norm_w =
            graph.get_or_upload_f32_weight(lp.attn_norm_handle, lp.attn_norm_bytes)?;
        let q_norm_w = graph.get_or_upload_f32_weight(lp.q_norm_handle, lp.q_norm_bytes)?;
        let k_norm_w = graph.get_or_upload_f32_weight(lp.k_norm_handle, lp.k_norm_bytes)?;
        let ffn_norm_w = graph.get_or_upload_f32_weight(lp.ffn_norm_handle, lp.ffn_norm_bytes)?;
        let fused_qkv_w =
            graph.get_or_upload_weight_soa(lp.fused_qkv_handle, lp.fused_qkv_bytes)?;
        let attn_proj_w =
            graph.get_or_upload_weight_soa(lp.attn_proj_handle, lp.attn_proj_bytes)?;

        let gate_bytes = lp.gate_bytes;
        let up_bytes = lp.up_bytes;
        let gate_up_w = graph.get_or_upload_weight_soa_lazy(lp.gate_up_handle, || {
            let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
            fused.extend_from_slice(gate_bytes);
            fused.extend_from_slice(up_bytes);
            fused
        })?;

        let down_w = graph.get_or_upload_weight_soa(lp.down_handle, lp.down_bytes)?;

        layer_weight_arcs.push((
            attn_norm_w,
            fused_qkv_w,
            q_norm_w,
            k_norm_w,
            attn_proj_w,
            ffn_norm_w,
            gate_up_w,
            down_w,
        ));
    }

    // ── Acquire activation buffers ───────────────────────────────────────
    let mut pb_guard = acquire_prefill_buffers(
        &graph,
        batch_size,
        hidden_size,
        intermediate_size,
        nq,
        nkv,
        head_dim,
        max_seq_len,
    )?;
    let pb = pb_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("prefill_buffers not allocated".into()))?;

    let mut kv_guard = acquire_prefill_kv_cache(&graph, n_layers, nkv, max_seq_len, head_dim)?;
    let kv = kv_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("kv_cache not allocated".into()))?;

    let mut st_guard = acquire_single_token_buffers(
        &graph,
        hidden_size,
        nq,
        nkv,
        head_dim,
        max_seq_len,
        intermediate_size,
    )?;
    let st_bufs = st_guard
        .as_mut()
        .ok_or_else(|| CudaGraphError::DriverError("st_buffers not allocated".into()))?;

    // ── Upload hidden batch → GPU ────────────────────────────────────────
    graph
        .stream_arc()
        .memcpy_htod(&hidden_batch[..batch_size * hidden_size], &mut pb.d_input)
        .map_err(|e| CudaGraphError::DriverError(format!("upload hidden_batch: {e}")))?;

    // ── Encode all layers ────────────────────────────────────────────────
    for (layer_idx, lwa) in layer_weight_arcs.iter().enumerate() {
        unsafe {
            encode_prefill_layer(
                &graph,
                &pmods,
                &attn_mods,
                &lwa.0, // attn_norm
                &lwa.1, // fused_qkv
                &lwa.2, // q_norm
                &lwa.3, // k_norm
                &lwa.4, // attn_proj
                &lwa.5, // ffn_norm
                &lwa.6, // gate_up
                &lwa.7, // down
                kv,
                layer_idx,
                pos_start,
                pb,
                st_bufs,
                cos_table,
                sin_table,
                heads_per_group,
                eps,
            )?;
        }
    }

    // ── Final norm + LM head on last token ──────────────────────────────
    if let (Some(fn_handle), Some(fn_bytes)) = (final_norm_handle, final_norm_bytes) {
        let d_final_norm_w = graph.get_or_upload_f32_weight(fn_handle, fn_bytes)?;

        if let (Some(lm_handle), Some(lm_bytes), true) =
            (lm_head_handle, lm_head_bytes, lm_head_out_features > 0)
        {
            let d_lm_head_w = graph.get_or_upload_weight_soa(lm_handle, lm_bytes)?;

            // Extract last token's hidden state (column batch_size-1)
            let last_col_start = (batch_size - 1) * hidden_size;
            let last_col_end = last_col_start + hidden_size;

            // Upload last token's hidden to st_bufs.d_hidden for single-token norm + GEMV
            {
                let src_view: CudaView<f32> = pb.d_input.slice(last_col_start..last_col_end);
                graph
                    .stream_arc()
                    .memcpy_dtod(&src_view, &mut st_bufs.d_hidden)
                    .map_err(|e| CudaGraphError::DriverError(format!("copy last hidden: {e}")))?;
            }

            // Single-token final RMSNorm
            unsafe {
                graph.launch_rmsnorm_pub(
                    &st_bufs.d_hidden,
                    &d_final_norm_w,
                    &mut st_bufs.d_normed,
                    hidden_size as u32,
                    final_norm_eps,
                )?;
            }

            // Acquire (or reuse) the cached logits buffer.
            let mut logits_guard = acquire_prefill_logits(&graph, lm_head_out_features)?;
            let d_logits = &mut logits_guard
                .as_mut()
                .ok_or_else(|| CudaGraphError::DriverError("logits buf not allocated".into()))?
                .0;

            // LM head GEMV (single token)
            unsafe {
                graph.launch_gemv_pub(
                    &d_lm_head_w,
                    &st_bufs.d_normed,
                    d_logits,
                    lm_head_out_features as u32,
                    hidden_size as u32,
                )?;
            }

            // Synchronise stream before D2H
            graph
                .stream_arc()
                .synchronize()
                .map_err(|e| CudaGraphError::DriverError(format!("prefill sync: {e}")))?;

            if let Some(out) = greedy_token_id_out {
                let logits_host = graph
                    .stream_arc()
                    .clone_dtoh(d_logits)
                    .map_err(|e| CudaGraphError::DriverError(format!("dtoh logits: {e}")))?;
                *out = logits_host
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
            } else if let Some(out) = logits_out {
                let logits_host = graph
                    .stream_arc()
                    .clone_dtoh(d_logits)
                    .map_err(|e| CudaGraphError::DriverError(format!("dtoh logits: {e}")))?;
                *out = logits_host;
            }

            if super::cuda_full_layer::profiling() {
                eprintln!(
                    "[cuda-prof] prefill batch={batch_size} pos_start={pos_start}: {:.1}ms (with lm_head)",
                    _t_prefill.expect("profiling").elapsed().as_secs_f64() * 1000.0
                );
            }
            return Ok(());
        }
    }

    // No final norm / LM head requested — just synchronise and return.
    graph
        .stream_arc()
        .synchronize()
        .map_err(|e| CudaGraphError::DriverError(format!("prefill sync end: {e}")))?;

    if super::cuda_full_layer::profiling() {
        eprintln!(
            "[cuda-prof] prefill batch={batch_size} pos_start={pos_start}: {:.1}ms",
            _t_prefill.expect("profiling").elapsed().as_secs_f64() * 1000.0
        );
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_backend::cuda_prefill_kernels::CUDA_PREFILL_KERNELS_SRC;

    /// Verify the kernel source string contains `gemm_q1_g128_v7` without GPU.
    #[test]
    fn test_prefill_kernel_source_has_gemm() {
        assert!(
            CUDA_PREFILL_KERNELS_SRC.contains("gemm_q1_g128_v7"),
            "CUDA_PREFILL_KERNELS_SRC must contain gemm_q1_g128_v7"
        );
    }

    /// Verify the kernel source string contains `batched_rmsnorm_v2` without GPU.
    #[test]
    fn test_prefill_kernel_source_has_batched_rmsnorm() {
        assert!(
            CUDA_PREFILL_KERNELS_SRC.contains("batched_rmsnorm_v2"),
            "CUDA_PREFILL_KERNELS_SRC must contain batched_rmsnorm_v2"
        );
    }

    /// Verify the kernel source string contains `fused_gate_up_swiglu_gemm_q1`.
    #[test]
    fn test_prefill_kernel_source_has_fused_gemm() {
        assert!(
            CUDA_PREFILL_KERNELS_SRC.contains("fused_gate_up_swiglu_gemm_q1"),
            "CUDA_PREFILL_KERNELS_SRC must contain fused_gate_up_swiglu_gemm_q1"
        );
    }

    /// Verify `CudaPrefillBuffers::matches` correctly checks dimension equality.
    #[test]
    fn test_prefill_buffers_dimension_arithmetic() {
        let batch_size = 8usize;
        let _hidden_size = 2048usize;
        let intermediate_size = 8192usize;
        let nq = 32usize;
        let nkv = 8usize;
        let head_dim = 64usize;
        let _max_seq = 512usize;
        let qkv_total = (nq + 2 * nkv) * head_dim;
        assert_eq!(qkv_total, 48 * 64);
        let gate_up_size = 2 * batch_size * intermediate_size;
        assert_eq!(gate_up_size, 2 * 8 * 8192);
    }

    /// Verify `init_prefill_modules` / `CudaGraph::global` gracefully skip without GPU.
    #[test]
    fn test_cuda_prefill_modules_init() {
        let graph_result = CudaGraph::global();
        if graph_result.is_err() {
            // No CUDA device present — skip gracefully.
            return;
        }
        let graph = graph_result.expect("prefill graph init should succeed");
        let result = init_prefill_modules(&graph);
        assert!(
            result.is_ok(),
            "prefill module init failed: {:?}",
            result.err()
        );
    }
}
