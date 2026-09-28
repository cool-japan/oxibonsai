//! Low-level CUDA kernel launchers for the full-layer attention pipeline.
//!
//! Each function wraps a single `CudaFunction` from [`CudaAttnModules`] with
//! the correct grid/block configuration and argument ordering. These are the
//! building blocks used by `encode_attn_phase` and friends in the parent
//! module.
//!
//! The cfg gate on the parent module (`native-cuda` + Linux/Windows) applies
//! here via module inclusion, so no additional `#[cfg(...)]` is needed.

use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};

use super::super::cuda_device_negotiation::{attn_scores_shared_bytes, ATTN_SCORES_BLOCK_DIM};
use super::super::cuda_graph::{CudaGraph, CudaGraphError};
use super::CudaAttnModules;

/// Launch `fused_qk_norm`.
///
/// Grid `(nq + nkv, 1, 1)`, block `(256, 1, 1)`.
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_fused_qk_norm(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_q_in: &CudaSlice<f32>,
    d_k_in: &CudaSlice<f32>,
    d_q_out: &mut CudaSlice<f32>,
    d_k_out: &mut CudaSlice<f32>,
    d_q_weight: &CudaSlice<f32>,
    d_k_weight: &CudaSlice<f32>,
    nq: u32,
    nkv: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (nq + nkv, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_qk_norm)
        .arg(d_q_in)
        .arg(d_k_in)
        .arg(d_q_out)
        .arg(d_k_out)
        .arg(d_q_weight)
        .arg(d_k_weight)
        .arg(&nq)
        .arg(&nkv)
        .arg(&head_dim)
        .arg(&eps)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_qk_norm launch: {e}")))
}

/// Launch `fused_qk_rope`.
///
/// Grid `(ceil(half_dim/64), nq + nkv, 1)`, block `(64, 1, 1)`.
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments, dead_code)]
pub(super) unsafe fn launch_fused_qk_rope(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_q_in: &CudaSlice<f32>,
    d_k_in: &CudaSlice<f32>,
    d_q_out: &mut CudaSlice<f32>,
    d_k_out: &mut CudaSlice<f32>,
    d_cos: &CudaSlice<f32>,
    d_sin: &CudaSlice<f32>,
    nq: u32,
    nkv: u32,
    half_dim: u32,
) -> Result<(), CudaGraphError> {
    let grid_x = half_dim.div_ceil(64);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, nq + nkv, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_qk_rope)
        .arg(d_q_in)
        .arg(d_k_in)
        .arg(d_q_out)
        .arg(d_k_out)
        .arg(d_cos)
        .arg(d_sin)
        .arg(&nq)
        .arg(&nkv)
        .arg(&half_dim)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_qk_rope launch: {e}")))
}

/// Launch `fused_qk_norm_rope`.
///
/// Grid `(nq + nkv, 1, 1)`, block `(256, 1, 1)`.
///
/// `d_k_in_view` is a `CudaView` pointing at the K section of the QKV buffer.
///
/// # Safety
/// All slices/views must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_qk_norm_rope(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_q_in: &CudaSlice<f32>,
    d_k_in_view: &CudaView<'_, f32>,
    d_q_out: &mut CudaSlice<f32>,
    d_k_out: &mut CudaSlice<f32>,
    d_q_weight: &CudaSlice<f32>,
    d_k_weight: &CudaSlice<f32>,
    d_cos: &CudaView<'_, f32>,
    d_sin: &CudaView<'_, f32>,
    nq: u32,
    nkv: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (nq + nkv, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_qk_norm_rope)
        .arg(d_q_in)
        .arg(d_k_in_view)
        .arg(d_q_out)
        .arg(d_k_out)
        .arg(d_q_weight)
        .arg(d_k_weight)
        .arg(d_cos)
        .arg(d_sin)
        .arg(&nq)
        .arg(&nkv)
        .arg(&head_dim)
        .arg(&eps)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_qk_norm_rope launch: {e}")))
}

/// Launch `fused_kv_store`.
///
/// Grid `(ceil(head_dim/64), nkv, 1)`, block `(64, 1, 1)`.
///
/// `d_pos_seqlen[0]` = current position (read by the kernel from device memory).
///
/// `layer_offset` is the `u64` element offset produced by
/// [`CudaKvCache::layer_offset_elements`](super::CudaKvCache::layer_offset_elements);
/// the kernel parameter is `unsigned long long` (finding **F4**).
///
/// **CUDA is unvalidated.** No CUDA hardware has run this launch; the
/// widened argument type has never been exercised by an actual `unsigned
/// long long` kernel launch.
///
/// # Safety
/// All slices/views must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_fused_kv_store(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_k_data: &CudaSlice<f32>,
    d_v_data_view: &CudaView<'_, f32>,
    d_k_cache: &mut CudaSlice<u16>,
    d_v_cache: &mut CudaSlice<u16>,
    head_dim: u32,
    nkv: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaView<'_, u32>,
    layer_offset: u64,
) -> Result<(), CudaGraphError> {
    let grid_x = head_dim.div_ceil(64);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, nkv, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.fused_kv_store)
        .arg(d_k_data)
        .arg(d_v_data_view)
        .arg(d_k_cache)
        .arg(d_v_cache)
        .arg(&head_dim)
        .arg(&nkv)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .arg(&layer_offset)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("fused_kv_store launch: {e}")))
}

/// Launch `batched_attn_scores_v2`.
///
/// Grid `(n_q, max_seq / BATCH_STRIDE, 1)`, block
/// `(ATTN_SCORES_BLOCK_DIM, 1, 1)`.
///
/// The grid Y dimension is fixed at `max_seq / BATCH_STRIDE` (not `seq_len`) so the
/// kernel sequence can be captured as a CUDA driver graph once and replayed for any
/// position.  Blocks with `pos_start >= seq_len` (read from `d_pos_seqlen[1]`) exit
/// immediately via the existing loop condition, adding only negligible overhead.
///
/// **Finding F3.** This is the one place that sizes the kernel's dynamic shared
/// memory, which holds the staged Q vector *and* the cross-warp partials. The
/// kernel used to declare a fixed `__shared__ float shared_q[128]` and stage one
/// element per thread, silently truncating the QK dot product to
/// `min(head_dim, 128)` — so Bonsai 2 27B (`head_dim = 256`) computed every
/// attention score from half its Q vector, with no error. `head_dim` above
/// [`CUDA_ATTN_MAX_HEAD_DIM`](super::super::cuda_device_negotiation::CUDA_ATTN_MAX_HEAD_DIM)
/// is now refused here rather than overrunning shared memory on the device.
///
/// # Errors
/// [`CudaGraphError::InvalidDimensions`] when `head_dim` is zero or wider
/// than the shared-memory budget allows (refused before any launch);
/// [`CudaGraphError::DriverError`] when the launch itself fails.
///
/// **CUDA is unvalidated.** No CUDA hardware has run this launch; the
/// shared-memory sizing and the precondition below are exercised only by
/// [`attn_scores_shared_bytes`]'s host-side unit tests, never by a real
/// launch.
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_batched_attn_scores_v2(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_queries: &CudaSlice<f32>,
    d_k_cache: &CudaSlice<u16>,
    d_scores: &mut CudaSlice<f32>,
    head_dim: u32,
    n_q: u32,
    n_kv: u32,
    heads_per_group: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaView<'_, u32>,
    inv_sqrt_hd: f32,
    cache_layer_offset: u64,
) -> Result<(), CudaGraphError> {
    const BATCH_STRIDE: u32 = 4;
    // F3 precondition: refuse before launch rather than corrupt shared memory.
    let shared_mem_bytes = attn_scores_shared_bytes(head_dim, ATTN_SCORES_BLOCK_DIM)
        .map_err(CudaGraphError::InvalidDimensions)?;
    // Fixed grid Y = max_seq / BATCH_STRIDE — constant across all decode positions,
    // allowing the kernel sequence to be captured as a replayable CUDA graph.
    let grid_y = max_seq.div_ceil(BATCH_STRIDE);
    let cfg = LaunchConfig {
        grid_dim: (n_q, grid_y, 1),
        block_dim: (ATTN_SCORES_BLOCK_DIM, 1, 1),
        shared_mem_bytes,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_attn_scores_v2)
        .arg(d_queries)
        .arg(d_k_cache)
        .arg(d_scores)
        .arg(&head_dim)
        .arg(&n_q)
        .arg(&n_kv)
        .arg(&heads_per_group)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .arg(&inv_sqrt_hd)
        .arg(&cache_layer_offset)
        .arg(&BATCH_STRIDE)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_attn_scores_v2 launch: {e}")))
}

/// Launch `batched_softmax`.
///
/// Grid `(n_q, 1, 1)`, block `(256, 1, 1)`.
///
/// `d_pos_seqlen[1]` = seq_len (read by the kernel from device memory).
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
pub(super) unsafe fn launch_batched_softmax(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_scores: &mut CudaSlice<f32>,
    n_q: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaView<'_, u32>,
) -> Result<(), CudaGraphError> {
    let cfg = LaunchConfig {
        grid_dim: (n_q, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_softmax)
        .arg(d_scores)
        .arg(&n_q)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_softmax launch: {e}")))
}

/// Launch `batched_attn_weighted_sum`.
///
/// Grid `(ceil(head_dim/64), n_q, 1)`, block `(64, 1, 1)`. The grid already
/// covers any `head_dim`, so this kernel needed no F3 change.
///
/// `d_pos_seqlen[1]` = seq_len (read by the kernel from device memory).
///
/// `cache_layer_offset` is the `u64` element offset from
/// [`CudaKvCache::layer_offset_elements`](super::CudaKvCache::layer_offset_elements)
/// (finding **F4**).
///
/// # Safety
/// All slices must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_batched_attn_weighted_sum(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_scores: &CudaSlice<f32>,
    d_v_cache: &CudaSlice<u16>,
    d_attn_out: &mut CudaSlice<f32>,
    head_dim: u32,
    n_q: u32,
    n_kv: u32,
    heads_per_group: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaView<'_, u32>,
    cache_layer_offset: u64,
) -> Result<(), CudaGraphError> {
    let grid_x = head_dim.div_ceil(64);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, n_q, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_attn_weighted_sum)
        .arg(d_scores)
        .arg(d_v_cache)
        .arg(d_attn_out)
        .arg(&head_dim)
        .arg(&n_q)
        .arg(&n_kv)
        .arg(&heads_per_group)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .arg(&cache_layer_offset)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| CudaGraphError::DriverError(format!("batched_attn_weighted_sum launch: {e}")))
}

/// [`launch_batched_attn_weighted_sum`], writing into a device **view**
/// rather than an owned `&mut CudaSlice<f32>` (finding **F9**'s last
/// per-token copy): lets a caller point the weighted-sum output directly at
/// a column of a larger batched buffer (e.g.
/// `pb.d_attn_out[t*nq*hd..(t+1)*nq*hd]`) without a `memcpy_dtod` afterward.
/// Same kernel, same launch configuration, same grid/block math as
/// [`launch_batched_attn_weighted_sum`] — only the destination argument's
/// type differs.
///
/// # Safety
/// All slices/views must be valid device pointers allocated on the graph's stream.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_batched_attn_weighted_sum_view(
    graph: &CudaGraph,
    mods: &CudaAttnModules,
    d_scores: &CudaSlice<f32>,
    d_v_cache: &CudaSlice<u16>,
    d_attn_out: &mut CudaViewMut<'_, f32>,
    head_dim: u32,
    n_q: u32,
    n_kv: u32,
    heads_per_group: u32,
    max_seq: u32,
    d_pos_seqlen: &CudaView<'_, u32>,
    cache_layer_offset: u64,
) -> Result<(), CudaGraphError> {
    let grid_x = head_dim.div_ceil(64);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, n_q, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    graph
        .stream_arc()
        .launch_builder(&mods.batched_attn_weighted_sum)
        .arg(d_scores)
        .arg(d_v_cache)
        .arg(d_attn_out)
        .arg(&head_dim)
        .arg(&n_q)
        .arg(&n_kv)
        .arg(&heads_per_group)
        .arg(&max_seq)
        .arg(d_pos_seqlen)
        .arg(&cache_layer_offset)
        .launch(cfg)
        .map(|_| ())
        .map_err(|e| {
            CudaGraphError::DriverError(format!("batched_attn_weighted_sum (view) launch: {e}"))
        })
}
